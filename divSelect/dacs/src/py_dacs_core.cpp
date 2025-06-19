#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

template <class ValueType, int _Cols=Eigen::Dynamic>
using rowvec_type = Eigen::Array<ValueType, 1, _Cols, Eigen::RowMajor>;
template <class ValueType, int _Rows=Eigen::Dynamic, int _Cols=Eigen::Dynamic>
using rowarr_type = Eigen::Array<ValueType, _Rows, _Cols, Eigen::RowMajor>;

template <class StateType, class XType>
void solve_pgd(
    StateType& state,
    XType& x
)
{
    using state_t = std::decay_t<StateType>;
    using index_t = typename state_t::index_t;
    using value_t = typename state_t::value_t;
    using rowvec_value_t = typename state_t::rowvec_value_t;

    const auto n = state.n;
    const auto max_iters = state.max_iters;
    const auto step0 = state.step;
    const auto theta0 = state.theta;
    const auto beta = state.beta;
    const auto abs_tol = state.abs_tol;
    const auto rel_tol = state.rel_tol;
    const auto debug = state.debug;
    auto& n_backtracks = state.n_backtracks;
    auto& iterates = state.iterates;

    // Initialize buffer.
    rowvec_value_t buffer(5*n);
    auto* buff_ptr = buffer.data();
    Eigen::Map<rowvec_value_t> x_prev(buff_ptr, n); buff_ptr += n;
    Eigen::Map<rowvec_value_t> y(buff_ptr, n); buff_ptr += n;
    Eigen::Map<rowvec_value_t> z(buff_ptr, n); buff_ptr += n;
    Eigen::Map<rowvec_value_t> grad_x(buff_ptr, n); buff_ptr += n;
    Eigen::Map<rowvec_value_t> grad_y(buff_ptr, n); buff_ptr += n;

    // Initialize data.
    x_prev = x;
    y = x;
    state.gradient(y, grad_y); 
    value_t theta = theta0;
    value_t obj_x_prev = state.objective(y, grad_y);

    for (index_t i = 0; i < max_iters; ++i) 
    {
        // Backtracking line search.
        value_t obj_x = NAN;
        {
            const value_t obj_y = state.objective(y, grad_y);
            value_t t = step0;
            index_t count = 0;
            while (1) {
                ++count;
                z = y - t * grad_y;
                state.prox(z, t, x); 
                state.gradient(x, grad_x);
                obj_x = state.objective(x, grad_x);

                // If prox operator is not changing much, stationary point probably.
                // For numerical stability, we should exit early here.
                const auto is_objective_converged = (
                    std::abs(obj_y - obj_x) <= abs_tol + rel_tol * std::abs(obj_y)
                );
                if (is_objective_converged) return;
                
                // If backtrack is successful, exit the loop.
                const auto is_backtrack_successful = (
                    (obj_x - obj_y) <= 
                    (
                        ((x - y) * grad_y).sum()
                        + 0.5 / t * (x - y).square().sum()
                    )
                );
                if (is_backtrack_successful) break;

                // Otherwise, backtrack failed and enough work to do.
                t *= beta;
            }
            n_backtracks.push_back(count);
            if (debug) {
                iterates.push_back(x);
            }
        }

        // Momentum update.
        const value_t theta_prev = theta;
        theta = (1 + std::sqrt(1 + 4 * theta_prev * theta_prev)) * 0.5;
        y = x + ((theta_prev - 1) / theta) * (x - x_prev);

        // Adaptive restart.
        if (((x - x_prev) * (y - x)).sum() > 0)
        {
            y = x;
            theta = theta0;
        }

        // Convergence criterion.
        if (
            (std::abs(obj_x_prev - obj_x) <= abs_tol + rel_tol * std::abs(obj_x_prev))
        ) {
            break;
        }

        // Update rest of the invariants.
        x_prev = x;
        state.gradient(y, grad_y);
        obj_x_prev = obj_x;
    }
}

template <
    class ValueType=double, 
    class IndexType=Eigen::Index
>
class PGDSolver
{
public:
    using value_t = ValueType;
    using index_t = IndexType;
    using rowvec_value_t = rowvec_type<value_t>;
    using rowvec_index_t = rowvec_type<index_t>;
    using rowarr_value_t = rowarr_type<value_t>;

    const index_t n;
    const value_t step;
    const value_t theta;
    const value_t beta;
    const index_t max_iters;
    const value_t abs_tol;
    const value_t rel_tol;
    const bool debug;

    std::vector<index_t> n_backtracks;
    std::vector<rowvec_value_t> iterates;

    explicit PGDSolver(
        index_t n,
        value_t step,
        value_t theta,
        value_t beta,
        size_t max_iters,
        value_t abs_tol,
        value_t rel_tol,
        bool debug
    ):
        n(n),
        step(step),
        theta(theta),
        beta(beta),
        max_iters(max_iters),
        abs_tol(abs_tol),
        rel_tol(rel_tol),
        debug(debug)
    {}
};

template <
    class ValueType=double, 
    class IndexType=Eigen::Index
>
class SharpePGDSolver: public PGDSolver<ValueType, IndexType>
{
public:
    using base_t = PGDSolver<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::rowvec_value_t;
    using typename base_t::rowvec_index_t;
    using typename base_t::rowarr_value_t;
    using base_t::n;

private:
    const Eigen::Map<const rowarr_value_t> _S;
    const value_t _C;

public:
    explicit SharpePGDSolver(
        const Eigen::Ref<const rowarr_value_t>& S,
        value_t C,
        value_t step,
        value_t theta,
        value_t beta,
        size_t max_iters,
        value_t abs_tol,
        value_t rel_tol,
        bool debug
    ):
        base_t(S.rows(), step, theta, beta, max_iters, abs_tol, rel_tol, debug),
        _S(S.data(), S.rows(), S.cols()),
        _C(C)
    {
        if (n != S.cols()) {
            throw std::runtime_error("S must be (n, n).");
        }
        if (C < 1.0/n) {
            throw std::runtime_error("C must be >= 1/n.");
        }
    }

    void prox(
        const Eigen::Ref<const rowvec_value_t>& y,
        value_t,
        Eigen::Ref<rowvec_value_t> x
    ) const
    {
        // Sort y.
        Eigen::Map<rowvec_value_t> y_sorted(x.data(), n);
        y_sorted = y;
        std::sort(
            y_sorted.data(), 
            y_sorted.data() + n
        );
        
        // Find optimal dual lmda.
        index_t z_idx = 0;
        value_t z_sum = 0.0;
        value_t lmda = 0.0;
        bool is_set = false;
        for (index_t i = 0; i < n; ++i) {
            const value_t y_i = y_sorted[i];
            const value_t t_i = _C * (n-i);
            value_t lower = (
                (i == 0) ?
                -std::numeric_limits<value_t>::infinity():
                y_sorted[i-1]
            );

            // Loop through all z's <= y_i.
            for (; (z_idx < n) && ((y_sorted[z_idx]-_C) <= y_i); ++z_idx) {
                const value_t upper = y_sorted[z_idx]-_C;
                lmda = (z_sum + t_i - 1) / (z_idx - i);
                if ((lower <= lmda) && (lmda <= upper)) {
                    is_set = true;
                    break;
                }
                z_sum += upper;
                lower = upper;
            }

            if (is_set) break;

            // Handle the last case when we must truncate to y_i.
            const value_t upper = y_i;
            lmda = (z_sum + t_i - 1) / (z_idx - i);
            if ((lower <= lmda) && (lmda <= upper)) {
                break;
            }
            lower = upper;

            z_sum -= y_i-_C;
        }

        // Clip and save.
        x = (y - lmda).min(_C).max(0);

        if (std::abs(x.sum() - 1) > 1e-9) {
            throw std::runtime_error(
                "Detected prox divergence. "
                "Sum of x is " + std::to_string(x.sum())
            );
        }
    }

    value_t objective(
        const Eigen::Ref<const rowvec_value_t>& x,
        const Eigen::Ref<const rowvec_value_t>& grad_x
    ) const
    {
        return 0.5 * (x * grad_x).sum();
    }

    void gradient(
        const Eigen::Ref<const rowvec_value_t>& x,
        Eigen::Ref<rowvec_value_t> out
    ) const
    {
        out.matrix() = x.matrix() * _S.matrix().transpose();
    }

    void solve(
        Eigen::Ref<rowvec_value_t> x
    ) 
    {
        solve_pgd(*this, x);
    }
};

template <
    class ValueType=double, 
    class IndexType=Eigen::Index
>
class MarkowitzPGDSolver: public PGDSolver<ValueType, IndexType>
{
public:
    using base_t = PGDSolver<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::rowvec_value_t;
    using typename base_t::rowvec_index_t;
    using typename base_t::rowarr_value_t;
    using base_t::n;
    using base_t::rel_tol;
    using base_t::max_iters;

private:
    const Eigen::Map<const rowarr_value_t> _S;
    const value_t _C;
    const value_t _gamma;

public:
    explicit MarkowitzPGDSolver(
        const Eigen::Ref<const rowarr_value_t>& S,
        value_t C,
        value_t gamma,
        value_t step,
        value_t theta,
        value_t beta,
        size_t max_iters,
        value_t abs_tol,
        value_t rel_tol,
        bool debug
    ):
        base_t(S.rows(), step, theta, beta, max_iters, abs_tol, rel_tol, debug),
        _S(S.data(), S.rows(), S.cols()),
        _C(C),
        _gamma(gamma)
    {
        if (n != S.cols()) {
            throw std::runtime_error("S must be (n, n).");
        }
        if (C < 0.0) {
            throw std::runtime_error("C must be >= 0.");
        }
    }

    void prox(
        const Eigen::Ref<const rowvec_value_t>& y,
        value_t,
        Eigen::Ref<rowvec_value_t> x
    ) const
    {
        // Handle special cases.
        if (_C <= 1.0/n) {
            x = std::max<value_t>(std::min<value_t>(y.mean(), 1), 0);
            return;
        }
        if (_C >= 1.0) {
            x = y.max(0).min(1);
            return;
        }

        // If y is already feasible, return it.
        const value_t y_sum = y.sum();
        if (
            (y.minCoeff() >= 0) &&
            (y.maxCoeff() <= std::min<value_t>(1, _C*y_sum))
        ) {
            x = y;
            return;
        }

        // Sort y.
        Eigen::Map<rowvec_value_t> y_sorted(x.data(), n);
        y_sorted = y;
        std::sort(
            y_sorted.data(), 
            y_sorted.data() + n
        );
        
        constexpr value_t inf = std::numeric_limits<value_t>::infinity();
        const value_t y2_sum = y.square().sum();
        value_t y_sum_i = 0; // sum of y until i-1.
        value_t best_obj = inf; // best objective
        value_t best_mu = NAN; // mu corresponding to the best objective
        value_t best_t = NAN; // t corresponding to the best objective

        // Loop through the intervals I_i = [y[i-1], y[i]] (y[-1] := -infty).
        for (index_t i = 0; i < n; ++i) {
            const value_t yi = y_sorted[i];
            const value_t li = (i == 0) ? -inf : y_sorted[i-1];
            const value_t ui = yi;

            // Loop through sub-intervals to split points that hit upper bound.
            const value_t nmi = n-i;
            const value_t denom_01 = _C * nmi - 1;
            const value_t denom_abs_01 = std::abs(denom_01);
            value_t y_sum_ik = 0; // sum of y from i to k (inclusive)
            value_t y2_sum_ik = 0; // sum of y**2 from i to k (inclusive)
            for (index_t k = i; k < n; ++k) {
                const value_t yk = y_sorted[k];
                if (yk > 1 + yi) break;

                y_sum_ik += yk;
                y2_sum_ik += yk * yk;
                
                // Compute lower and upper bounds on u(t) + mu(t).
                const value_t lik = yk;
                const value_t uik = (k == n-1) ? inf : y_sorted[k+1];

                // Set up some useful quantities.
                const value_t nmkm1 = n-k-1;
                const value_t denom_00 = _C * nmkm1 - 1;
                const value_t denom_abs_00 = std::abs(denom_00);
                const value_t kmip1 = k-i+1;
                const value_t kmip1_li = kmip1 * li;
                const value_t kmip1_ui = kmip1 * ui;
                const value_t kmip1_lik = kmip1 * lik;
                const value_t kmip1_uik = kmip1 * uik;

                // Compute bounds from case of t in [0, 1/C].
                const value_t numer_lower_00 = kmip1_li - y_sum_ik;
                const value_t t_lower_00 = (
                    (denom_abs_00 <= 0) ? 
                    ((numer_lower_00 <= 0) ? -inf : inf) : 
                    numer_lower_00 / denom_abs_00
                );
                const value_t numer_upper_00 = kmip1_ui - y_sum_ik;
                const value_t t_upper_00 = (
                    (denom_abs_00 <= 0) ?
                    ((numer_upper_00 >= 0) ? inf : -inf) :
                    numer_upper_00 / denom_abs_00
                );
                const value_t numer_lower_01 = kmip1_lik - y_sum_ik;
                const value_t t_lower_01 = (
                    (denom_abs_01 <= 0) ?
                    ((numer_lower_01 <= 0) ? -inf : inf) : 
                    numer_lower_01 / denom_abs_01
                );
                const value_t numer_upper_01 = kmip1_uik - y_sum_ik;
                const value_t t_upper_01 = (
                    (denom_abs_01 <= 0) ?
                    ((numer_upper_01 >= 0) ? -inf : inf) : 
                    numer_upper_01 / denom_abs_01
                );
                const value_t t_lower_0 = std::max<value_t>(
                    std::max<value_t>(
                        (denom_00 < 0) ? -t_upper_00 : t_lower_00,
                        (denom_01 < 0) ? -t_upper_01 : t_lower_01
                    ),
                    0.0
                );
                const value_t t_upper_0 = std::min<value_t>(
                    std::min<value_t>(
                        (denom_00 < 0) ? -t_lower_00 : t_upper_00,
                        (denom_01 < 0) ? -t_lower_01 : t_upper_01
                    ),
                    1.0/_C
                );
                const value_t y_sum_k = y_sum - y_sum_i - y_sum_ik;
                const value_t denom_00_over_kmip1 = denom_00 / kmip1;
                const value_t t_unc_0 = (
                    (_C * y_sum_k - denom_00_over_kmip1 * y_sum_ik) /
                    (_C + denom_00 * (_C + denom_00_over_kmip1))
                );
                const value_t t_0 = std::max<value_t>(
                    t_lower_0,
                    std::min<value_t>(t_unc_0, t_upper_0)
                );
                const value_t mu_0 = (y_sum_ik + denom_00 * t_0) / kmip1;
                const value_t u_0 = _C * t_0;
                const value_t obj_0 = (
                    // If t is not feasible, mark this as impossible.
                    (t_lower_0 > t_upper_0) ? inf :
                    (
                        y2_sum - y2_sum_ik
                        + mu_0 * mu_0 * kmip1
                        + u_0 * (u_0 * nmkm1 - 2 * y_sum_k)
                    )
                );

                // Compute absolute bounds from case of t in [1/C, n].
                const value_t t_lower_10 = nmkm1 - numer_upper_00;
                const value_t t_upper_10 = nmkm1 - numer_lower_00;
                const value_t t_lower_11 = nmi - numer_upper_01;
                const value_t t_upper_11 = nmi - numer_lower_01;
                const value_t t_lower_1 = std::max<value_t>(
                    std::max<value_t>(t_lower_10, t_lower_11),
                    1.0/_C
                );
                const value_t t_upper_1 = std::min<value_t>(
                    std::min<value_t>(t_upper_10, t_upper_11),
                    n
                );
                const value_t t_unc_1 = y_sum_ik + nmkm1;
                const value_t t_1 = std::max<value_t>(
                    t_lower_1, 
                    std::min<value_t>(t_unc_1, t_upper_1)
                );
                const value_t mu_1 = (t_unc_1 - t_1) / kmip1;
                const value_t obj_1 = (
                    // If t is not feasible, mark this as impossible.
                    (t_lower_1 > t_upper_1) ? inf :
                    (
                        y2_sum - y2_sum_ik
                        + mu_1 * mu_1 * kmip1
                        + nmkm1 - 2.0 * y_sum_k
                    )
                );

                // Compare objectives.
                const value_t obj = (obj_0 < obj_1) ? obj_0 : obj_1;
                const value_t mu = (obj_0 < obj_1) ? mu_0 : mu_1;
                const value_t t = (obj_0 < obj_1) ? t_0 : t_1;
                if (obj < best_obj) {
                    best_obj = obj;
                    best_mu = mu;
                    best_t = t;
                }
            }

            y_sum_i += ui;
        }

        if (y2_sum < best_obj) {
            x.setZero();
            return;
        }

        // Store the final solution.
        x = (y - best_mu).max(0).min(std::min<value_t>(_C * best_t, 1.0));

        if (std::abs(x.sum() - best_t) > 1e-9) {
            throw std::runtime_error(
                "Detected prox divergence. "
                "Sum of x is " + std::to_string(x.sum()) + ". "
                "t is " + std::to_string(best_t)
            );
        }
    }

    value_t objective(
        const Eigen::Ref<const rowvec_value_t>& x,
        const Eigen::Ref<const rowvec_value_t>& grad_x
    ) const
    {
        return 0.5 * ((x * grad_x).sum() - x.sum());
    }

    void gradient(
        const Eigen::Ref<const rowvec_value_t>& x,
        Eigen::Ref<rowvec_value_t> out
    ) const
    {
        out = _gamma * (x.matrix() * _S.matrix().transpose()).array() - 1;
    }

    void solve(
        Eigen::Ref<rowvec_value_t> x
    ) 
    {
        solve_pgd(*this, x);
    }
};

template <class ValueType, class IndexType>
static void
add_pgd_solver(py::module_& m, const char* name)
{
    using internal_t = PGDSolver<ValueType, IndexType>;
    using value_t = typename internal_t::value_t;
    using index_t = typename internal_t::index_t;
    using rowarr_value_t = typename internal_t::rowarr_value_t;
    using rowvec_index_t = typename internal_t::rowvec_index_t;
    py::class_<internal_t>(m, name)
        .def(py::init<
            index_t,
            value_t,
            value_t,
            value_t,
            size_t,
            value_t,
            value_t,
            bool
        >(),
            py::arg("n"),
            py::arg("step"),
            py::arg("theta"),
            py::arg("beta"),
            py::arg("max_iters"),
            py::arg("abs_tol"),
            py::arg("rel_tol"),
            py::arg("debug")
        )
        .def_readonly("n", &internal_t::n)
        .def_property_readonly("n_backtracks", [&](const internal_t& s) {
            return Eigen::Map<const rowvec_index_t>(
                s.n_backtracks.data(),
                s.n_backtracks.size()
            );
        })
        .def_property_readonly("iterates", [&](const internal_t& s) {
            rowarr_value_t iterates(s.iterates.size(), s.n);
            for (size_t i = 0; i < s.iterates.size(); ++i) {
                iterates.row(i) = s.iterates[i];
            }
            return iterates;
        })
        ;
}

template <class ValueType, class IndexType>
static void 
add_sharpe_pgd_solver(py::module_& m, const char* name)
{
    using internal_t = SharpePGDSolver<ValueType, IndexType>;
    using base_t = typename internal_t::base_t;
    using value_t = typename internal_t::value_t;
    using rowarr_value_t = typename internal_t::rowarr_value_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<
            const Eigen::Ref<const rowarr_value_t>&,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            value_t,
            value_t,
            bool
        >(),
            py::arg("S").noconvert(),
            py::arg("C"),
            py::arg("step"),
            py::arg("theta"),
            py::arg("beta"),
            py::arg("max_iters"),
            py::arg("abs_tol"),
            py::arg("rel_tol"),
            py::arg("debug")
        )
        .def("prox", &internal_t::prox)
        .def("objective", &internal_t::objective)
        .def("gradient", &internal_t::gradient)
        .def("solve", &internal_t::solve)
        ;
}

template <class ValueType, class IndexType>
static void 
add_markowitz_pgd_solver(py::module_& m, const char* name)
{
    using internal_t = MarkowitzPGDSolver<ValueType, IndexType>;
    using base_t = typename internal_t::base_t;
    using value_t = typename internal_t::value_t;
    using rowarr_value_t = typename internal_t::rowarr_value_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<
            const Eigen::Ref<const rowarr_value_t>&,
            value_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            value_t,
            value_t,
            bool
        >(),
            py::arg("S").noconvert(),
            py::arg("C"),
            py::arg("gamma"),
            py::arg("step"),
            py::arg("theta"),
            py::arg("beta"),
            py::arg("max_iters"),
            py::arg("abs_tol"),
            py::arg("rel_tol"),
            py::arg("debug")
        )
        .def("prox", &internal_t::prox)
        .def("objective", &internal_t::objective)
        .def("gradient", &internal_t::gradient)
        .def("solve", &internal_t::solve)
        ;
}

template <class ValueType=double, class IndexType=Eigen::Index>
ValueType mc_sharpe(
    const Eigen::Ref<const rowarr_type<ValueType>>& S,
    const Eigen::Ref<const rowarr_type<IndexType>>& B
)
{
    using value_t = ValueType;
    using index_t = IndexType;

    value_t sum = 0;
    for (index_t i = 0; i < B.rows(); ++i) {
        const auto b = B.row(i);

        // Compute b^T S b assuming b is 0 or 1.
        value_t diag_sum = 0;
        value_t off_diag_sum = 0;
        for (index_t j = 0; j < b.size(); ++j) {
            const auto bj = b[j];
            if (bj == 0) continue;
            diag_sum += S(j,j);
            for (index_t k = 0; k < j; ++k) {
                if (b[k] == 0) continue;
                off_diag_sum += S(j,k);
            }
        }
        value_t denom = 2 * off_diag_sum + diag_sum;
        denom = std::sqrt(std::max<value_t>(denom, 0.0));
        sum += b.sum() / (denom + (denom <= 0));
    }
    return sum / B.rows();
}

template <class ValueType=double, class IndexType=Eigen::Index>
rowarr_type<ValueType> subset(
    const Eigen::Ref<const rowarr_type<ValueType>>& S,
    const Eigen::Ref<const rowvec_type<IndexType>>& indices
)
{
    using index_t = IndexType;

    const auto s = indices.size();
    rowarr_type<ValueType> out(s, s);

    for (index_t i = 0; i < s; ++i) {
        const auto ii = indices[i];
        for (index_t j = 0; j <= i; ++j) {
            const auto jj = indices[j];
            out(i,j) = S(ii, jj);
        }
    }

    for (index_t i = 0; i < s; ++i) {
        for (index_t j = i+1; j < s; ++j) {
            out(i,j) = out(j,i);
        }
    }

    return out;
}

PYBIND11_MODULE(dacs_core, m) {
    add_pgd_solver<double, Eigen::Index>(m, "PGDSolver");
    add_sharpe_pgd_solver<double, Eigen::Index>(m, "SharpePGDSolver");
    add_markowitz_pgd_solver<double, Eigen::Index>(m, "MarkowitzPGDSolver");

    m.def("mc_sharpe", mc_sharpe<>);
    m.def("subset", subset<>);
}
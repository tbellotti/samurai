// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <samurai/samurai.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"

#include <chrono>

double eps_g = 5.e-5, eps_f = 1e-1;

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

template <class Config>
auto init(samurai::Mesh<Config> &mesh)
{
    samurai::BC<2> bc{ {{ {samurai::BCType::dirichlet, 0},
                       {samurai::BCType::dirichlet, 0},
                       {samurai::BCType::neumann, 0},
                       {samurai::BCType::neumann, 0}
                    }} };
    samurai::Field<Config> u{"u", mesh, bc};
    u.array().fill(0);

    // mesh.for_each_cell([&](auto &cell) {
    //     auto center = cell.center();
    //     auto x = center[0];
    //     auto y = center[1];
    //     u[cell] = exp(-20 * (x * x + y * y));
    // });

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        double radius = .1;
        double x_center = 0.5, y_center = 0.5;
        if (((center[0] - x_center) * (center[0] - x_center) +
                (center[1] - y_center) * (center[1] - y_center))
                <= radius * radius)
            u[cell] = 1;
        else
            u[cell] = 0;


        double x_center2 = 0.2, y_center2 = 0.2;
        if (((center[0] - x_center2) * (center[0] - x_center2) +
                (center[1] - y_center2) * (center[1] - y_center2))
                <= radius * radius)
            u[cell] = -1;

    });

    // mesh.for_each_cell([&](auto &cell) {
    //     auto center = cell.center();
    //     double theta = M_PI / 4;
    //     auto x = cos(theta) * center[0] - sin(theta) * center[1];
    //     auto y = sin(theta) * center[0] + cos(theta) * center[1];
    //     double x_corner = -0.1;
    //     double y_corner = -0.1;
    //     double length = 0.2;

    //     if ((x_corner <= x) and (x <= x_corner + length) and
    //         (y_corner <= y) and (y <= y_corner + length))
    //         u[cell] = 1;
    //     else
    //         u[cell] = 0;
    // });

    return u;
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 2;
    using Config = samurai::MRConfig<dim>;
    using interval_t = typename Config::interval_t;

    std::size_t min_level = 2, max_level = 8; // For the moment, we do not adapt the mesh.
    // samurai::Box<double, dim> box({-2, -2}, {2, 2});
    samurai::Box<double, dim> box({0, 0}, {1, 1});
    samurai::Mesh<Config> mesh{box, min_level, max_level};

    // For this, we take a unit vector
    std::array<double, 2> k{{sqrt(2.0)/2.0, sqrt(2.0)/2.0}};
    double dt = .05/(1<<max_level);

    auto u = init(mesh);

    spdlog::set_level(spdlog::level::warn);

    for (std::size_t nt=0; nt<500; ++nt)
    {


        tic();
        std::stringstream s;
        s << "VF_burgers_2d_ite_" << nt;
        auto h5file = samurai::Hdf5(s.str().data());
        h5file.add_mesh(mesh);
        samurai::Field<Config> level_{"level", mesh};
        mesh.for_each_cell([&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
        h5file.add_field(u);
        h5file.add_field(level_);
        auto duration = toc();
        std::cout << "save: " << duration << "s\n";
        std::cout << "iteration " << nt << "\n";
        tic();


        for (std::size_t i=0; i<max_level-min_level; ++i)
        {
            if (coarsening(u, i, nt))
                break;
        }
        duration = toc();
        std::cout << "coarsening: " << duration << "s\n";

        tic();
        for (std::size_t i=0; i<max_level-min_level; ++i)
        {
            if (refinement(u, i, nt))
                break;
        }
        duration = toc();
        std::cout << "refinement: " << duration << "s\n";

        samurai::mr_projection(u);
        samurai::amr_prediction(u);
        u.update_bc();

        tic();
        samurai::Field<Config> unp1{"u", mesh};

        //unp1 = u - dt * samurai::upwind_scalar_burgers(k, u);
        unp1 = u - dt * samurai::upwind_scalar_burgers(k, u);

        for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
        {
            xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

            stencil = {{-1, 0}};

            auto subset_right = intersection(translate(mesh[samurai::MeshType::cells][level+1], stencil),
                                             mesh[samurai::MeshType::cells][level])
                               .on(level);

            subset_right([&](auto& index, auto& interval, auto)
            {
                auto i = interval[0];
                auto j = index[0];
                double dx = 1./(1<<level);

                unp1(level, i, j) = unp1(level, i, j) + dt/dx * (samurai::upwind_scalar_burgers_op<interval_t>(level, i, j).right_flux(k, u)
                                                                - .5*samurai::upwind_scalar_burgers_op<interval_t>(level+1, 2*i+1, 2*j).right_flux(k, u)
                                                                - .5*samurai::upwind_scalar_burgers_op<interval_t>(level+1, 2*i+1, 2*j+1).right_flux(k, u));
            });

            stencil = {{1, 0}};

            auto subset_left = intersection(translate(mesh[samurai::MeshType::cells][level+1], stencil),
                                            mesh[samurai::MeshType::cells][level])
                               .on(level);

            subset_left([&](auto& index, auto& interval, auto)
            {
                auto i = interval[0];
                auto j = index[0];
                double dx = 1./(1<<level);

                unp1(level, i, j) = unp1(level, i, j) - dt/dx * (samurai::upwind_scalar_burgers_op<interval_t>(level, i, j).left_flux(k, u)
                                                              - .5 * samurai::upwind_scalar_burgers_op<interval_t>(level+1, 2*i, 2*j).left_flux(k, u)
                                                              - .5 * samurai::upwind_scalar_burgers_op<interval_t>(level+1, 2*i, 2*j+1).left_flux(k, u));
            });

            stencil = {{0, -1}};

            auto subset_up = intersection(translate(mesh[samurai::MeshType::cells][level+1], stencil),
                                       mesh[samurai::MeshType::cells][level])
                               .on(level);

            subset_up([&](auto& index, auto& interval, auto)
            {
                auto i = interval[0];
                auto j = index[0];
                double dx = 1./(1<<level);

                unp1(level, i, j) = unp1(level, i, j) + dt/dx * (samurai::upwind_scalar_burgers_op<interval_t>(level, i, j).up_flux(k, u)
                                                              - .5 * samurai::upwind_scalar_burgers_op<interval_t>(level+1, 2*i, 2*j+1).up_flux(k, u)
                                                              - .5 * samurai::upwind_scalar_burgers_op<interval_t>(level+1, 2*i+1, 2*j+1).up_flux(k, u));
            });

            stencil = {{0, 1}};

            auto subset_down = intersection(translate(mesh[samurai::MeshType::cells][level+1], stencil),
                                       mesh[samurai::MeshType::cells][level])
                               .on(level);

            subset_down([&](auto& index, auto& interval, auto)
            {
                auto i = interval[0];
                auto j = index[0];
                double dx = 1./(1<<level);

                unp1(level, i, j) = unp1(level, i, j) - dt/dx * (samurai::upwind_scalar_burgers_op<interval_t>(level, i, j).down_flux(k, u)
                                                                - .5 * samurai::upwind_scalar_burgers_op<interval_t>(level+1, 2*i, 2*j).down_flux(k, u)
                                                                - .5 * samurai::upwind_scalar_burgers_op<interval_t>(level+1, 2*i+1, 2*j).down_flux(k, u));
            });

        }

        std::swap(u.array(), unp1.array());

        duration = toc();
        std::cout << "upwind: " << duration << "s\n";


    }
    return 0;
}
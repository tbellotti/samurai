// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fakhari, Geier, Lee 
// A mass-conserving lattice Boltzmann method with dynamic grid refinement for immiscible two-phase flows

#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <samurai/mr/adapt.hpp>
#include <samurai/mr/coarsening.hpp>
#include <samurai/mr/criteria.hpp>
#include <samurai/mr/harten.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/refinement.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/statistics.hpp>

#include "prediction_map_2d.hpp"
#include "boundary_conditions.hpp"

#include "utils_lbm_mr_2d.hpp"

double rho_in = 1.;
double rho_out = 1.5;
double nu_in = 5.e-6;
double nu_out = nu_in;
double sigma = 1.e-3;
double grav = 0.01;//0.01;
double p_ref = 10.;

template<class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0, j = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    auto rotation_of_pi_over_two = [] (int alpha, int k, int h)
    {
        // Returns the rotation of (k, h) of an angle alpha * pi / 2.
        // All the operations are performed on integer, to be exact
        int cosinus = static_cast<int>(std::round(std::cos(alpha * M_PI / 2.)));
        int sinus   = static_cast<int>(std::round(std::sin(alpha * M_PI / 2.)));

        return std::pair<int, int> (cosinus * k - sinus   * h,
                                      sinus * k + cosinus * h);
    };

    // Transforms the coordinates to apply the rotation
    auto tau = [] (int delta, int k)
    {
        // The case in which delta = 0 is rather exceptional
        if (delta == 0) {
            return k;
        }
        else {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < tmp) ? (k - tmp) : (k - tmp + 1));
        }
    };

    auto tau_inverse = [] (int delta, int k)
    {
        if (delta == 0) {
            return k;
        }
        else
        {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < 0) ? (k + tmp) : (k + tmp - 1));
        }
    };

    for(std::size_t k = 0; k < max_level - min_level + 1; ++k)
    {
        int size = (1<<k);

        // We have 9 velocity out of which 8 are moving
        // 4 are moving along the axis, thus needing only 2 fluxes each (entering-exiting)
        // and 4 along the diagonals, thus needing  6 fluxes

        // 4 * 2 + 4 * 6 = 8 + 24 = 32
        data[k].resize(32);

        // Parallel velocities
        for (int alpha = 0; alpha <= 3; ++alpha)
        {
            for (int l = 0; l < size; ++l)
            {
                // The reference direction from which the other ones are computed is that of (1, 0)
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i   * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1)* size - 1), tau(k, j * size + l));

                data[k][0 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_in.first ), tau_inverse(k, rotated_in.second ));
                data[k][1 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second));
            }
        }

        // Diagonal velocities

        // Translation of the indices from which we start saving the new computations
        int offset = 4 * 2;
        for (int alpha = 0; alpha <= 3; ++alpha)
        {

            // First side
            for (int l = 0; l < size - 1; ++l)
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i   * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1)* size - 1), tau(k, j * size + l));

                data[k][offset + 6 * alpha + 0] += prediction(k, tau_inverse(k, rotated_in.first ),  tau_inverse(k, rotated_in.second ));
                data[k][offset + 6 * alpha + 3] += prediction(k, tau_inverse(k, rotated_out.first),  tau_inverse(k, rotated_out.second));

            }
            // Cell on the diagonal
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i    * size - 1), tau(k,  j    * size - 1));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1) * size - 1), tau(k, (j+1) * size - 1));

                data[k][offset + 6 * alpha + 1] += prediction(k, tau_inverse(k, rotated_in.first ),  tau_inverse(k, rotated_in.second ));
                data[k][offset + 6 * alpha + 4] += prediction(k, tau_inverse(k, rotated_out.first),  tau_inverse(k, rotated_out.second));

            }
            // Second side
            for (int l = 0; l < size - 1; ++l)
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha,  tau(k, i*size + l), tau(k,  j    * size - 1));
                auto rotated_out = rotation_of_pi_over_two(alpha,  tau(k, i*size + l), tau(k, (j+1) * size - 1));

                data[k][offset + 6 * alpha + 2] += prediction(k, tau_inverse(k, rotated_in.first ),  tau_inverse(k, rotated_in.second ));
                data[k][offset + 6 * alpha + 5] += prediction(k, tau_inverse(k, rotated_out.first),  tau_inverse(k, rotated_out.second));
            }
        }
    }
    return data;
}


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

double vel_x(const double u0, double x, double y)
{
    return -u0 * std::pow(std::sin(M_PI*x), 2.) * std::sin(2.*M_PI*y);
}

double vel_y(const double u0, double x, double y)
{
    return u0 * std::pow(std::sin(M_PI*y), 2.) * std::sin(2.*M_PI*x);
}

template<class Config>
auto init_vel(samurai::MRMesh<Config> &mesh, const double u0)
{
    using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;

    auto vel = samurai::make_field<double, 2>("vel", mesh);
    vel.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        vel[cell][0] = vel_x(u0, x, y);
        vel[cell][1] = vel_y(u0, x, y);
    });

    return vel;
}

template<class Config, class VelField>
auto init_f(samurai::MRMesh<Config> &mesh, VelField & u, double lambda, double mob, double W)
{
    constexpr std::size_t nvel = 18;
    using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;

    auto f = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);


    auto phi_field = samurai::make_field<double, 1>("phi", mesh);
    phi_field.fill(0);

    auto pressure_field = samurai::make_field<double, 1>("pressure", mesh);
    pressure_field.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double xi = std::sqrt(std::pow(x - .5, 2.) + 2.*std::pow(y - .5, 2.)) - 0.15;
        // double xi = y - 0.5;

        phi_field[cell] = .5 - .5 * std::tanh(2.*xi/W);

    });

    // Tres crade 
    std::size_t n_cells = (1 << mesh.max_level());
    double dx = 1./(1 << mesh.max_level());

    for (auto i = 0; i < n_cells; ++i)  {
        double press = p_ref;
        for (auto j = 0; j < n_cells; ++j)  {
            double rho = rho_out + ((phi_field(mesh.max_level(), {i, i+1}, j)[0]) * (rho_in - rho_out)); 
            // press += dx * grav * rho;
            pressure_field(mesh.max_level(), {i, i+1}, j) = press;
        }  
    }


    auto level = mesh.max_level();
    auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                        mesh[mesh_id_t::cells][level]);

    double cs = lambda/std::sqrt(3.);

    double r1 = 1.0 / lambda;
    double r2 = 1.0 / (lambda*lambda);
    double r3 = 1.0 / (lambda*lambda*lambda);
    double r4 = 1.0 / (lambda*lambda*lambda*lambda);

    leaves([&](auto& interval, auto& index) {
        auto k = interval; // Logical index in x
        auto h = index[0]; // Logical index in y

        // // Bonne facon
        auto diff_x = xt::eval( 3./(2.*dx)*( 1./9 *(phi_field(level, k + 1, h    ) - phi_field(level, k - 1, h    ))
                                            -1./9 *(phi_field(level, k - 1, h    ) - phi_field(level, k + 1, h    ))
                                            +1./36*(phi_field(level, k + 1, h + 1) - phi_field(level, k - 1, h - 1))
                                            -1./36*(phi_field(level, k - 1, h + 1) - phi_field(level, k + 1, h - 1))
                                            -1./36*(phi_field(level, k - 1, h - 1) - phi_field(level, k + 1, h + 1))
                                            +1./36*(phi_field(level, k + 1, h - 1) - phi_field(level, k - 1, h + 1))));

        auto diff_y = xt::eval( 3./(2.*dx)*( 1./9 *(phi_field(level, k    , h + 1) - phi_field(level, k    , h - 1))
                                            -1./9 *(phi_field(level, k    , h - 1) - phi_field(level, k    , h + 1))
                                            +1./36*(phi_field(level, k + 1, h + 1) - phi_field(level, k - 1, h - 1))
                                            +1./36*(phi_field(level, k - 1, h + 1) - phi_field(level, k + 1, h - 1))
                                            -1./36*(phi_field(level, k - 1, h - 1) - phi_field(level, k + 1, h + 1))
                                            -1./36*(phi_field(level, k + 1, h - 1) - phi_field(level, k - 1, h + 1))));

        // Since we are divising by the modulus, we have to truncate if it
        // is too small (division by 0)
        auto abs_diff = xt::eval(xt::sqrt(xt::pow(diff_x, 2.) + xt::pow(diff_y, 2.)));

        // This is not with the right orientation but it is ok
        // for the equation, because it is coherent with the choice
        // of initial datum.
        diff_x =  diff_x / xt::maximum(abs_diff, 1.e-13);
        diff_y =  diff_y / xt::maximum(abs_diff, 1.e-13);

        // Now diff_x and diff_y are the components of the normal vector
        // along the axis

        auto phi_loc = xt::eval(phi_field(level, k, h));
        auto theta = mob/(W*cs*cs) * (1. - 4.*xt::pow(1 - phi_loc - .5, 2.));
        // auto theta = mob/(W*cs*cs) * (1. - 4.*xt::pow(phi_loc - .5, 2.));

        auto Vx = xt::eval(0.*u(0, level, k, h));
        auto Vy = xt::eval(0.*u(1, level, k, h));

        auto V_abs_sq = xt::pow(Vx, 2.) + xt::pow(Vy, 2.);

        auto m0 = phi_loc;
        auto m1 = Vx*m0 + cs*cs*theta*diff_x; 
        auto m2 = Vy*m0 + cs*cs*theta*diff_y;
        auto m3 = -2.*m0/r2+3.*V_abs_sq*m0;
        auto m4 = -Vx/r2*m0 - 1./3 * std::pow(lambda, 4.)*theta*diff_x ;             
        auto m5 = -Vy/r2*m0 - 1./3 * std::pow(lambda, 4.)*theta*diff_y ;             
        auto m6 = m0/r4-3.*V_abs_sq/r2*m0;
        auto m7 = Vx*Vx*m0-Vy*Vy*m0;        
        auto m8 = Vx*Vy*m0;  

        f(0, level, k, h) = xt::eval((1./9)*m0                                  -  (1./9)*r2*m3                                +   (1./9)*r4*m6                         ); 
        f(1, level, k, h) = xt::eval((1./9)*m0   + (1./6)*r1*m1                 - (1./36)*r2*m3 - (1./6)*r3*m4                 -  (1./18)*r4*m6 + .25*r2*m7             ); 
        f(2, level, k, h) = xt::eval((1./9)*m0                  +  (1./6)*r1*m2 - (1./36)*r2*m3                -  (1./6)*r3*m5 -  (1./18)*r4*m6 - .25*r2*m7             ); 
        f(3, level, k, h) = xt::eval((1./9)*m0   - (1./6)*r1*m1                 - (1./36)*r2*m3 + (1./6)*r3*m4                 -  (1./18)*r4*m6 + .25*r2*m7             ); 
        f(4, level, k, h) = xt::eval((1./9)*m0                  -  (1./6)*r1*m2 - (1./36)*r2*m3                +  (1./6)*r3*m5 -  (1./18)*r4*m6 - .25*r2*m7             ); 
        f(5, level, k, h) = xt::eval((1./9)*m0   + (1./6)*r1*m1 +  (1./6)*r1*m2 + (1./18)*r2*m3 +(1./12)*r3*m4 + (1./12)*r3*m5 +  (1./36)*r4*m6             + .25*r2*m8 ); 
        f(6, level, k, h) = xt::eval((1./9)*m0   - (1./6)*r1*m1 +  (1./6)*r1*m2 + (1./18)*r2*m3 -(1./12)*r3*m4 + (1./12)*r3*m5 +  (1./36)*r4*m6             - .25*r2*m8 ); 
        f(7, level, k, h) = xt::eval((1./9)*m0   - (1./6)*r1*m1 -  (1./6)*r1*m2 + (1./18)*r2*m3 -(1./12)*r3*m4 - (1./12)*r3*m5 +  (1./36)*r4*m6             + .25*r2*m8 ); 
        f(8, level, k, h) = xt::eval((1./9)*m0   + (1./6)*r1*m1 -  (1./6)*r1*m2 + (1./18)*r2*m3 +(1./12)*r3*m4 - (1./12)*r3*m5 +  (1./36)*r4*m6             - .25*r2*m8 ); 

        // auto pressure = xt::eval(pressure_field(level, k, h));
        auto rho = xt::eval(rho_out + phi_loc * (rho_in - rho_out)); 
        auto pressure = xt::eval(rho * cs*cs);

        f(0 + 9, level, k, h) = (4./ 9*.5*(rho_in+rho_out));
        f(1 + 9, level, k, h) = (1./ 9*.5*(rho_in+rho_out));
        f(2 + 9, level, k, h) = (1./ 9*.5*(rho_in+rho_out));
        f(3 + 9, level, k, h) = (1./ 9*.5*(rho_in+rho_out));
        f(4 + 9, level, k, h) = (1./ 9*.5*(rho_in+rho_out));
        f(5 + 9, level, k, h) = (1./36*.5*(rho_in+rho_out));
        f(6 + 9, level, k, h) = (1./36*.5*(rho_in+rho_out));
        f(7 + 9, level, k, h) = (1./36*.5*(rho_in+rho_out));
        f(8 + 9, level, k, h) = (1./36*.5*(rho_in+rho_out));
    });

    return f;
}


template<class Field, class Pred, class Func>
void one_time_step(Field &f, const Pred & pred_coeff, Func&& update_bc_for_level, const double lambda, const double s, const double mob, const double W, const double u0, const double t, const double T)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::interval_t::coord_index_t;
    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();
    
    // update_bc_for_level(f, level);

    Field advected{"advected", mesh};
    advected.array().fill(0.);
    Field fluxes{"fluxes", mesh};  // This stored the fluxes computed at the level of the overleaves
    fluxes.array().fill(0.);

    samurai::mr_projection(f);
    for (std::size_t level = min_level - 1; level <= max_level; ++level)
    {
        update_bc_for_level(f, level); // It is important to do so
    }
    samurai::mr_prediction(f, update_bc_for_level);
    samurai::mr_prediction_overleaves(f, update_bc_for_level);

    std::size_t shift = 9;

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        if (level == max_level) {
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                mesh[mesh_id_t::cells][level]);

            // We do the advection
            leaves([&](auto& interval, auto& index) {
                auto k = interval; // Logical index in x
                auto h = index[0]; // Logical index in y

                advected(0, level, k, h) =  f(0, level, k    , h    );
                advected(1, level, k, h) =  f(1, level, k - 1, h    );
                advected(2, level, k, h) =  f(2, level, k    , h - 1);
                advected(3, level, k, h) =  f(3, level, k + 1, h    );
                advected(4, level, k, h) =  f(4, level, k    , h + 1);
                advected(5, level, k, h) =  f(5, level, k - 1, h - 1);
                advected(6, level, k, h) =  f(6, level, k + 1, h - 1);
                advected(7, level, k, h) =  f(7, level, k + 1, h + 1);
                advected(8, level, k, h) =  f(8, level, k - 1, h + 1);

                advected(0 + shift, level, k, h) =  f(0 + shift, level, k    , h    );
                advected(1 + shift, level, k, h) =  f(1 + shift, level, k - 1, h    );
                advected(2 + shift, level, k, h) =  f(2 + shift, level, k    , h - 1);
                advected(3 + shift, level, k, h) =  f(3 + shift, level, k + 1, h    );
                advected(4 + shift, level, k, h) =  f(4 + shift, level, k    , h + 1);
                advected(5 + shift, level, k, h) =  f(5 + shift, level, k - 1, h - 1);
                advected(6 + shift, level, k, h) =  f(6 + shift, level, k + 1, h - 1);
                advected(7 + shift, level, k, h) =  f(7 + shift, level, k + 1, h + 1);
                advected(8 + shift, level, k, h) =  f(8 + shift, level, k - 1, h + 1);
            });
        }
        
    }


    // We are ready to collide

    // We construct the conserved momentum to easily compute
    // Its gradient giving the normal vector
    auto phi = samurai::make_field<double, 1>("phi", mesh);
    phi.fill(0);
    for (std::size_t level = 0; level <= max_level; ++level)    {

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                        mesh[mesh_id_t::cells][level]);

        leaves([&](auto& interval, auto& index) {
            auto k = interval; // Logical index in x
            auto h = index[0]; // Logical index in y

            phi(level, k, h) = xt::eval(advected(0, level, k, h)
                                       +advected(1, level, k, h)+advected(2, level, k, h)+advected(3, level, k, h)+advected(4, level, k, h)
                                       +advected(5, level, k, h)+advected(6, level, k, h)+advected(7, level, k, h)+advected(8, level, k, h));
        });
    }


    samurai::mr_projection(phi);
    for (std::size_t level = min_level - 1; level <= max_level; ++level)
    {
        update_bc_for_level(phi, level); // It is important to do so
    }
    samurai::mr_prediction(phi, update_bc_for_level);

    // LOOK IF WE HAVE TO UPDATE BC FOR THE FIELD.
    double cs = lambda/std::sqrt(3.);

    double l1 = lambda;
    double l2 = l1 * lambda;
    double l3 = l2 * lambda;
    double l4 = l3 * lambda;

    double r1 = 1.0 / lambda;
    double r2 = 1.0 / (lambda*lambda);
    double r3 = 1.0 / (lambda*lambda*lambda);
    double r4 = 1.0 / (lambda*lambda*lambda*lambda);

    double beta = 12.*sigma/W;
    double kappa = 3./2*sigma*W;

    for (std::size_t level = 0; level <= max_level; ++level)    {
        
        double dx = 1./(1 << level);

        double dt = dx / lambda;


        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);
        leaves([&](auto& interval, auto& index) {
            auto k = interval; // Logical index in x
            auto h = index[0]; // Logical index in y

            // // We compute the normal vector
            // // Calcul bete
            // auto diff_x = xt::eval((phi(level, k + 1, h    ) - phi(level, k - 1, h    ))/(2.*dx));
            // auto diff_y = xt::eval((phi(level, k    , h + 1) - phi(level, k    , h - 1))/(2.*dx));

            // Bonne facon
            auto diff_x = xt::eval( 3./(2.*dx)*( 1./9 *(phi(level, k + 1, h    ) - phi(level, k - 1, h    ))
                                                -1./9 *(phi(level, k - 1, h    ) - phi(level, k + 1, h    ))
                                                +1./36*(phi(level, k + 1, h + 1) - phi(level, k - 1, h - 1))
                                                -1./36*(phi(level, k - 1, h + 1) - phi(level, k + 1, h - 1))
                                                -1./36*(phi(level, k - 1, h - 1) - phi(level, k + 1, h + 1))
                                                +1./36*(phi(level, k + 1, h - 1) - phi(level, k - 1, h + 1))));

            auto diff_y = xt::eval( 3./(2.*dx)*( 1./9 *(phi(level, k    , h + 1) - phi(level, k    , h - 1))
                                                -1./9 *(phi(level, k    , h - 1) - phi(level, k    , h + 1))
                                                +1./36*(phi(level, k + 1, h + 1) - phi(level, k - 1, h - 1))
                                                +1./36*(phi(level, k - 1, h + 1) - phi(level, k + 1, h - 1))
                                                -1./36*(phi(level, k - 1, h - 1) - phi(level, k + 1, h + 1))
                                                -1./36*(phi(level, k + 1, h - 1) - phi(level, k - 1, h + 1))));

            auto ddphi = xt::eval((phi(level, k + 1, h    ) - 2.*phi(level, k    , h    ) + phi(level, k - 1, h    )
                                  +phi(level, k    , h + 1) - 2.*phi(level, k    , h    ) + phi(level, k    , h - 1))/(dx*dx));

            // Since we are divising by the modulus, we have to truncate if it
            // is too small (division by 0)
            auto abs_diff = xt::eval(xt::sqrt(xt::pow(diff_x, 2.) + xt::pow(diff_y, 2.)));

            // This is not with the right orientation but it is ok
            // for the equation, because it is coherent with the choice
            // of initial datum.
            auto n_x =  diff_x / xt::maximum(abs_diff, 1.e-14);
            auto n_y =  diff_y / xt::maximum(abs_diff, 1.e-14);

            // Now diff_x and diff_y are the components of the normal vector
            // along the axis

            auto phi_loc = xt::eval(phi(level, k, h));
            auto theta = mob/(W*cs*cs) * (1. - 4.*xt::pow(1 - phi_loc - .5, 2.));

            auto rho = rho_out + (rho_in - rho_out) * phi_loc; // According to our choice for phi (1 - phi in the paper)
            // The chemical potential has wholly changed its sign, but like the normal, I do not think that we should fix it
            // auto chem_pot = xt::eval(.0*(4.*beta*(phi_loc)*(phi_loc-1.)*(phi_loc-.5) - kappa*ddphi));
            auto chem_pot = xt::eval((4.*beta*(phi_loc)*(phi_loc-1.)*(phi_loc-.5) - kappa*ddphi));

            auto grav_y = xt::eval(grav*(rho-rho_out));

            auto ux = xt::eval(1./rho*(lambda*( advected(1+shift, level, k, h)-advected(3+shift, level, k, h)
                                                +advected(5+shift, level, k, h)-advected(6+shift, level, k, h)-advected(7+shift, level, k, h)+advected(8+shift, level, k, h))
                                                +.5*dt*chem_pot*diff_x));

            auto uy = xt::eval(1./rho*(lambda*( advected(2+shift, level, k, h)-advected(4+shift, level, k, h)
                                                +advected(5+shift, level, k, h)+advected(6+shift, level, k, h)-advected(7+shift, level, k, h)-advected(8+shift, level, k, h))
                                                +.5*dt*(chem_pot*diff_y + grav_y)));


            auto abs_u_sq = xt::eval(xt::pow(ux, 2.) + xt::pow(uy, 2.));


            auto rho0 = xt::eval(advected(0+shift, level, k, h)+advected(1+shift, level, k, h)+advected(2+shift, level, k, h)+advected(3+shift, level, k, h)+advected(4+shift, level, k, h)
                                                               +advected(5+shift, level, k, h)+advected(6+shift, level, k, h)+advected(7+shift, level, k, h)+advected(8+shift, level, k, h) 
                                                +.5*dt*(rho_in-rho_out)*(ux*diff_x + uy*diff_y));

            auto m0 = xt::eval(       advected(0, level, k, h) +   advected(1, level, k, h) +   advected(2, level, k, h)  +  advected(3, level, k, h)  +   advected(4, level, k, h) +   advected(5, level, k, h) +   advected(6, level, k, h) +   advected(7, level, k, h) +   advected(8, level, k, h)) ;
            auto m1 = xt::eval(l1*(                                advected(1, level, k, h)                               -  advected(3, level, k, h)                               +   advected(5, level, k, h) -   advected(6, level, k, h) -   advected(7, level, k, h) +   advected(8, level, k, h)));
            auto m2 = xt::eval(l1*(                                                             advected(2, level, k, h)                               -   advected(4, level, k, h) +   advected(5, level, k, h) +   advected(6, level, k, h) -   advected(7, level, k, h) -   advected(8, level, k, h)));
            auto m3 = xt::eval(l2*(-4*advected(0, level, k, h) -   advected(1, level, k, h) -   advected(2, level, k, h)  -   advected(3, level, k, h) -   advected(4, level, k, h) + 2*advected(5, level, k, h) + 2*advected(6, level, k, h) + 2*advected(7, level, k, h) + 2*advected(8, level, k, h)));
            auto m4 = xt::eval(l3*(                            - 2*advected(1, level, k, h)                               + 2*advected(3, level, k, h)                              +   advected(5, level, k, h) -   advected(6, level, k, h)   - advected(7, level, k, h) +   advected(8, level, k, h)));
            auto m5 = xt::eval(l3*(                                                         - 2*advected(2, level, k, h)                               + 2*advected(4, level, k, h)   + advected(5, level, k, h) +   advected(6, level, k, h)   - advected(7, level, k, h) -   advected(8, level, k, h)));
            auto m6 = xt::eval(l4*( 4*advected(0, level, k, h) - 2*advected(1, level, k, h) - 2*advected(2, level, k, h)  - 2*advected(3, level, k, h) - 2*advected(4, level, k, h)   + advected(5, level, k, h) +   advected(6, level, k, h)   + advected(7, level, k, h) +   advected(8, level, k, h)));
            auto m7 = xt::eval(l2*(                                advected(1, level, k, h) -   advected(2, level, k, h)  +   advected(3, level, k, h) -   advected(4, level, k, h)                            ));
            auto m8 = xt::eval(l2*(                                                                                                                                                     advected(5, level, k, h) -   advected(6, level, k, h) +   advected(7, level, k, h) -   advected(8, level, k, h)));

            auto so = 1.;

            m1 = (1. - s ) * m1 + s  * (ux*m0 + cs*cs*theta*n_x);
            m2 = (1. - s ) * m2 + s  * (uy*m0 + cs*cs*theta*n_y);
            m3 = (1. - so) * m3 + so * (-2.*m0/r2+3.*abs_u_sq*m0);
            m4 = (1. - so) * m4 + so * (-ux/r2*m0 - 1./3 * std::pow(lambda, 4.)*theta*n_x);
            m5 = (1. - so) * m5 + so * (-uy/r2*m0 - 1./3 * std::pow(lambda, 4.)*theta*n_y);
            m6 = (1. - so) * m6 + so * (m0/r4-3.*abs_u_sq/r2*m0);
            m7 = (1. - so) * m7 + so * (ux*ux*m0-uy*uy*m0);
            m8 = (1. - so) * m8 + so * (ux*uy*m0);

            f(0, level, k, h) = (1./9)*m0                                  -  (1./9)*r2*m3                                +   (1./9)*r4*m6                         ;
            f(1, level, k, h) = (1./9)*m0   + (1./6)*r1*m1                 - (1./36)*r2*m3 - (1./6)*r3*m4                 -  (1./18)*r4*m6 + .25*r2*m7             ;
            f(2, level, k, h) = (1./9)*m0                  +  (1./6)*r1*m2 - (1./36)*r2*m3                -  (1./6)*r3*m5 -  (1./18)*r4*m6 - .25*r2*m7             ;
            f(3, level, k, h) = (1./9)*m0   - (1./6)*r1*m1                 - (1./36)*r2*m3 + (1./6)*r3*m4                 -  (1./18)*r4*m6 + .25*r2*m7             ;
            f(4, level, k, h) = (1./9)*m0                  -  (1./6)*r1*m2 - (1./36)*r2*m3                +  (1./6)*r3*m5 -  (1./18)*r4*m6 - .25*r2*m7             ;
            f(5, level, k, h) = (1./9)*m0   + (1./6)*r1*m1 +  (1./6)*r1*m2 + (1./18)*r2*m3 +(1./12)*r3*m4 + (1./12)*r3*m5 +  (1./36)*r4*m6             + .25*r2*m8 ;
            f(6, level, k, h) = (1./9)*m0   - (1./6)*r1*m1 +  (1./6)*r1*m2 + (1./18)*r2*m3 -(1./12)*r3*m4 + (1./12)*r3*m5 +  (1./36)*r4*m6             - .25*r2*m8 ;
            f(7, level, k, h) = (1./9)*m0   - (1./6)*r1*m1 -  (1./6)*r1*m2 + (1./18)*r2*m3 -(1./12)*r3*m4 - (1./12)*r3*m5 +  (1./36)*r4*m6             + .25*r2*m8 ;
            f(8, level, k, h) = (1./9)*m0   + (1./6)*r1*m1 -  (1./6)*r1*m2 + (1./18)*r2*m3 +(1./12)*r3*m4 - (1./12)*r3*m5 +  (1./36)*r4*m6             - .25*r2*m8 ;


            // Let us go to the NS part

            auto mx_0 = xt::eval(4./ 9*(1                                                                            -abs_u_sq/(2.*cs*cs))); 
            auto mx_1 = xt::eval(1./ 9*(1+lambda*ux/(cs*cs)      +xt::pow(+lambda  *ux    , 2.)/(2.*std::pow(cs, 4.))-abs_u_sq/(2.*cs*cs))); 
            auto mx_2 = xt::eval(1./ 9*(1+lambda*uy/(cs*cs)      +xt::pow(+lambda  *uy    , 2.)/(2.*std::pow(cs, 4.))-abs_u_sq/(2.*cs*cs))); 
            auto mx_3 = xt::eval(1./ 9*(1-lambda*ux/(cs*cs)      +xt::pow(-lambda  *ux    , 2.)/(2.*std::pow(cs, 4.))-abs_u_sq/(2.*cs*cs))); 
            auto mx_4 = xt::eval(1./ 9*(1-lambda*uy/(cs*cs)      +xt::pow(-lambda  *uy    , 2.)/(2.*std::pow(cs, 4.))-abs_u_sq/(2.*cs*cs))); 
            auto mx_5 = xt::eval(1./36*(1+lambda*( ux+uy)/(cs*cs)+xt::pow( lambda*( ux+uy), 2.)/(2.*std::pow(cs, 4.))-abs_u_sq/(2.*cs*cs))); 
            auto mx_6 = xt::eval(1./36*(1+lambda*(-ux+uy)/(cs*cs)+xt::pow( lambda*(-ux+uy), 2.)/(2.*std::pow(cs, 4.))-abs_u_sq/(2.*cs*cs))); 
            auto mx_7 = xt::eval(1./36*(1+lambda*(-ux-uy)/(cs*cs)+xt::pow( lambda*(-ux-uy), 2.)/(2.*std::pow(cs, 4.))-abs_u_sq/(2.*cs*cs))); 
            auto mx_8 = xt::eval(1./36*(1+lambda*( ux-uy)/(cs*cs)+xt::pow( lambda*( ux-uy), 2.)/(2.*std::pow(cs, 4.))-abs_u_sq/(2.*cs*cs))); 

            auto eq_0 = xt::eval(rho*(mx_0 - 4./ 9) + 4./ 9*rho0); 
            auto eq_1 = xt::eval(rho*(mx_1 - 1./ 9) + 1./ 9*rho0); 
            auto eq_2 = xt::eval(rho*(mx_2 - 1./ 9) + 1./ 9*rho0); 
            auto eq_3 = xt::eval(rho*(mx_3 - 1./ 9) + 1./ 9*rho0); 
            auto eq_4 = xt::eval(rho*(mx_4 - 1./ 9) + 1./ 9*rho0); 
            auto eq_5 = xt::eval(rho*(mx_5 - 1./36) + 1./36*rho0); 
            auto eq_6 = xt::eval(rho*(mx_6 - 1./36) + 1./36*rho0); 
            auto eq_7 = xt::eval(rho*(mx_7 - 1./36) + 1./36*rho0); 
            auto eq_8 = xt::eval(rho*(mx_8 - 1./36) + 1./36*rho0);    

            double srel = 1.;         

            auto source_0 = xt::eval((1. - .5*srel)*dt/(cs*cs)*((cs*cs*(rho_in-rho_out)*(mx_0 - 4./ 9)+chem_pot*mx_0)*((       -ux)*diff_x + (       -uy)*diff_y) + mx_0*grav_y*(       -uy)));
            auto source_1 = xt::eval((1. - .5*srel)*dt/(cs*cs)*((cs*cs*(rho_in-rho_out)*(mx_1 - 1./ 9)+chem_pot*mx_1)*(( lambda-ux)*diff_x + (       -uy)*diff_y) + mx_1*grav_y*(       -uy)));
            auto source_2 = xt::eval((1. - .5*srel)*dt/(cs*cs)*((cs*cs*(rho_in-rho_out)*(mx_2 - 1./ 9)+chem_pot*mx_2)*((       -ux)*diff_x + ( lambda-uy)*diff_y) + mx_2*grav_y*( lambda-uy)));
            auto source_3 = xt::eval((1. - .5*srel)*dt/(cs*cs)*((cs*cs*(rho_in-rho_out)*(mx_3 - 1./ 9)+chem_pot*mx_3)*((-lambda-ux)*diff_x + (       -uy)*diff_y) + mx_3*grav_y*(       -uy)));
            auto source_4 = xt::eval((1. - .5*srel)*dt/(cs*cs)*((cs*cs*(rho_in-rho_out)*(mx_4 - 1./ 9)+chem_pot*mx_4)*((       -ux)*diff_x + (-lambda-uy)*diff_y) + mx_4*grav_y*(-lambda-uy)));
            auto source_5 = xt::eval((1. - .5*srel)*dt/(cs*cs)*((cs*cs*(rho_in-rho_out)*(mx_5 - 1./36)+chem_pot*mx_5)*(( lambda-ux)*diff_x + ( lambda-uy)*diff_y) + mx_5*grav_y*( lambda-uy)));
            auto source_6 = xt::eval((1. - .5*srel)*dt/(cs*cs)*((cs*cs*(rho_in-rho_out)*(mx_6 - 1./36)+chem_pot*mx_6)*((-lambda-ux)*diff_x + ( lambda-uy)*diff_y) + mx_6*grav_y*( lambda-uy)));
            auto source_7 = xt::eval((1. - .5*srel)*dt/(cs*cs)*((cs*cs*(rho_in-rho_out)*(mx_7 - 1./36)+chem_pot*mx_7)*((-lambda-ux)*diff_x + (-lambda-uy)*diff_y) + mx_7*grav_y*(-lambda-uy)));
            auto source_8 = xt::eval((1. - .5*srel)*dt/(cs*cs)*((cs*cs*(rho_in-rho_out)*(mx_8 - 1./36)+chem_pot*mx_8)*(( lambda-ux)*diff_x + (-lambda-uy)*diff_y) + mx_8*grav_y*(-lambda-uy)));

            f(0+shift, level, k, h) = xt::eval((1. - srel) * f(0+shift, level, k, h) + srel * eq_0 + source_0);
            f(1+shift, level, k, h) = xt::eval((1. - srel) * f(1+shift, level, k, h) + srel * eq_1 + source_1);
            f(2+shift, level, k, h) = xt::eval((1. - srel) * f(2+shift, level, k, h) + srel * eq_2 + source_2);
            f(3+shift, level, k, h) = xt::eval((1. - srel) * f(3+shift, level, k, h) + srel * eq_3 + source_3);
            f(4+shift, level, k, h) = xt::eval((1. - srel) * f(4+shift, level, k, h) + srel * eq_4 + source_4);
            f(5+shift, level, k, h) = xt::eval((1. - srel) * f(5+shift, level, k, h) + srel * eq_5 + source_5);
            f(6+shift, level, k, h) = xt::eval((1. - srel) * f(6+shift, level, k, h) + srel * eq_6 + source_6);
            f(7+shift, level, k, h) = xt::eval((1. - srel) * f(7+shift, level, k, h) + srel * eq_7 + source_7);
            f(8+shift, level, k, h) = xt::eval((1. - srel) * f(8+shift, level, k, h) + srel * eq_8 + source_8);

        });
    }        
}

template<class Field, class VelField>
void save_solution(Field &f, VelField & u, double eps, std::size_t ite, double mob, double W, double t, std::string ext="")
{
    using value_t = typename Field::value_type;

    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D2Q9_diphasic_test_1_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto phi = samurai::make_field<value_t, 1>("phi", mesh);
    auto rho = samurai::make_field<value_t, 1>("rho", mesh);

    auto rho0 = samurai::make_field<value_t, 1>("rho0", mesh);
    auto ux = samurai::make_field<value_t, 1>("ux", mesh);
    auto uy = samurai::make_field<value_t, 1>("uy", mesh);

    auto level_ = samurai::make_field<double, 1>("level", mesh);
   
    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        phi[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3] + f[cell][4]
                               + f[cell][5] + f[cell][6] + f[cell][7] + f[cell][8];
        
        rho[cell] = rho_out + (rho_in - rho_out) * phi[cell]; 


        rho0[cell] = f[cell][9] + f[cell][10] + f[cell][11] + f[cell][12] + f[cell][13]
                             + f[cell][14] + f[cell][15] + f[cell][16] + f[cell][17];

        ux[cell] =  (  f[cell][10]               - f[cell][12]
                    + f[cell][14] - f[cell][15] - f[cell][16] + f[cell][17])/rho[cell];

        uy[cell] =  (              + f[cell][11]               - f[cell][13]
                    + f[cell][14] + f[cell][15] - f[cell][16] - f[cell][17])/rho[cell];




        level_[cell] = static_cast<double>(cell.level);
    });



    samurai::save(str.str().data(), mesh, phi, rho, rho0, f, ux, uy);
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d2q4_3_Euler",
                             "Multi resolution for a D2Q4 LBM scheme for the scalar advection equation");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("7"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.0001"))
                       ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                       ("ite", "number of iteration", cxxopts::value<std::size_t>()->default_value("100"))
                       ("reg", "regularity", cxxopts::value<double>()->default_value("0."))
                       ("config", "Lax-Liu configuration", cxxopts::value<int>()->default_value("12"))
                       ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {
            std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn},
                                                               {"info", spdlog::level::info}};
            constexpr size_t dim = 2;
            using Config = samurai::MRConfig<dim, 2>;


            // double mobility = .001;
            double lambda = 1.;
            std::size_t min_level = 8;
            std::size_t max_level = 8;

            samurai::Box<double, dim> box({0, 0}, {1, 1});
            samurai::MRMesh<Config> mesh(box, min_level, max_level);
            using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;
            using coord_index_t = typename samurai::MRMesh<Config>::coord_index_t;

            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            double W = 5. * dx;
            double mobility = .1 * lambda * dx;
            double u0 = 0.1*lambda;

            double Pe = u0 * W / mobility;

            // double W = 5.*dx;

            double s = 1./(.5 + 3.*mobility/(lambda*dx));

            // Initialization
            auto u     = init_vel(mesh, u0); 
            auto f     = init_f(mesh, u, lambda, mobility, W); 


            std::cout<<std::endl<<"s = "<<s<<"   Peclet = "<<Pe<<std::endl;

            auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);

            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_D2Q4_3_Euler_constant_extension(field, level);
            };

            double t = 0.;
            double T = 20.;
            std::size_t N = static_cast<std::size_t>(T / dt);

            int howoften = 32;
            double epsilon = 0.001;
            // auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);


            for (std::size_t nb_ite = 0; nb_ite <= N; ++nb_ite)
            {
                std::cout<<"Iteration = "<<nb_ite<<std::endl;
                // MRadaptation(epsilon, 10.);

                // save_normal(f, 0, nb_ite);
                if (nb_ite%howoften == 0)
                    save_solution(f, u, 0, nb_ite/howoften, mobility, W, t);

                one_time_step(f, pred_coeff, update_bc_for_level, lambda, s, mobility, W, u0, t, T);

                t += dt;
            }

                  
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}

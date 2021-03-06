// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

auto generate_mesh(std::size_t start_level, std::size_t max_level)
{
    constexpr std::size_t dim = 2;
    samurai::Box<int, dim> box({0, 0}, {1<<start_level, 1<<start_level});
    samurai::CellArray<dim> ca;

    ca[start_level] = {start_level, box};

    for(std::size_t ite = 0; ite < max_level - start_level; ++ite)
    {
        samurai::CellList<dim> cl;

        samurai::for_each_interval(ca, [&](std::size_t level, const auto& interval, const auto& index)
        {
            auto choice = xt::random::choice(xt::xtensor_fixed<bool, xt::xshape<2>>{true, false}, interval.size());
            for(int i = interval.start, ic = 0; i<interval.end; ++i, ++ic)
            {
                if (choice[ic])
                {
                    cl[level + 1][2*index].add_interval({2*i, 2*i+2});
                    cl[level + 1][2*index + 1].add_interval({2*i, 2*i+2});
                }
                else
                {
                    cl[level][index].add_point(i);
                }
            }
        });

        ca = {cl, true};
    }

    return ca;
}

int main()
{
    constexpr std::size_t dim = 2;
    std::size_t start_level = 1;
    std::size_t max_refinement_level = 7;
    auto ca = generate_mesh(start_level, max_refinement_level);

    std::size_t min_level = ca.min_level();
    std::size_t max_level = ca.max_level();

    samurai::save("mesh_before", ca);
    // xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 1}, {-1, -1}, {-1, 1}, {1, -1}};
    while(true)
    {
        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for(std::size_t level = min_level + 2; level <= max_level; ++level)
        {
            for(std::size_t level_below = min_level; level_below < level - 1; ++level_below)
            {
                for(std::size_t i = 0; i < stencil.shape()[0]; ++i)
                {
                    auto s = xt::view(stencil, i);
                    auto set = samurai::intersection(samurai::translate(ca[level], s), ca[level_below]).on(level_below);
                    set([&](const auto& i, const auto& index)
                    {
                        tag(level_below, i, index[0]) = true;
                    });
                }
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_cell(ca, [&](auto cell)
        {
            auto i = cell.indices[0];
            auto j = cell.indices[1];
            if (tag[cell])
            {
                cl[cell.level + 1][{2*j}].add_interval({2*i, 2*i+2});
                cl[cell.level + 1][{2*j + 1}].add_interval({2*i, 2*i+2});
            }
            else
            {
                cl[cell.level][{j}].add_point(i);
            }
        });
        samurai::CellArray<dim> new_ca = {cl, true};

        if(new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }
    samurai::save("mesh_after", ca);

    return 0;
}
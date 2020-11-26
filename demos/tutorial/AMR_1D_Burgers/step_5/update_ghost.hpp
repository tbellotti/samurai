#pragma once

#include <samurai/algorithm.hpp>
#include <samurai/subset/subset_op.hpp>

template<class Field>
void update_ghost(Field& phi)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    auto mesh = phi.mesh();

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    /**
     *
     *   |------|------|
     *
     *   |=============|-------------|
     */
    for (std::size_t level = max_level; level >= min_level; --level)
    {
        auto expr = samurai::intersection(mesh[mesh_id_t::cells][level],
                                          mesh[mesh_id_t::cells_and_ghosts][level - 1])
                   .on(level - 1);

        expr([&](const auto& i, auto)
        {
            phi(level - 1, i) = 0.5*(phi(level, 2*i) + phi(level, 2*i + 1));
        });
    }

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto expr = samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh.domain())
                   .on(level);

        expr([&](const auto& i, auto)
        {
            phi(level, i) = 0.;
        });
    }

    /**
     *
     *          |======|------|------|
     *
     *   |-------------|
     */
    for (std::size_t level = min_level + 1; level <= max_level; ++level)
    {
        auto expr = samurai::intersection(mesh.domain(),
                                          samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level],
                                                              mesh.get_union()[level]))
                   .on(level);

        expr([&](const auto& i, auto)
        {
            auto i_coarse = i >> 1;
            if (i.start & 1)
            {
                phi(level, i) = phi(level - 1, i_coarse) + 1./8*(phi(level - 1, i_coarse + 1) - phi(level - 1, i_coarse - 1));
            }
            else
            {
                phi(level, i) = phi(level - 1, i_coarse) - 1./8*(phi(level - 1, i_coarse + 1) - phi(level - 1, i_coarse - 1));
            }
        });
    }
}
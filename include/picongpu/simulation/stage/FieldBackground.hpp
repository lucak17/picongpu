/* Copyright 2013-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/fields/background/cellwiseOperation.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>


namespace picongpu
{
namespace simulation
{
namespace stage
{

    //! Functor for the stage of the PIC loop applying field background
    class FieldBackground
    {
    public:

        /** Create a field background functor
         *
         * Having this in constructor is a temporary solution.
         *
         * @param cellDescription mapping for kernels
         */
        FieldBackground( MappingDesc const cellDescription ):
            cellDescription( cellDescription )
        {
        }

        /** Add the field background to the current density
         *
         * @tparam T_Functor functor type compatible to nvidia::functors
         *
         * @param step index of time iteration
         * @param functor functor to apply to the background
         */
        template< typename T_Functor >
        void operator( )( uint32_t const step, T_Functor functor ) const
        {
            using namespace pmacc;
            DataConnector & dc = Environment< >::get( ).DataConnector( );
            auto fieldE = dc.get< FieldE >( FieldE::getName( ), true );
            auto fieldB = dc.get< FieldB >( FieldB::getName( ), true );
            using Background = cellwiseOperation::CellwiseOperation<
                CORE + BORDER + GUARD
            >;
            Background background( cellDescription );
            background(
                fieldE,
                functor,
                FieldBackgroundE( fieldE->getUnit( ) ),
                step,
                FieldBackgroundE::InfluenceParticlePusher
            );
            background(
                fieldB,
                functor,
                FieldBackgroundB( fieldB->getUnit( ) ),
                step,
                FieldBackgroundB::InfluenceParticlePusher
            );
            dc.releaseData( FieldE::getName( ) );
            dc.releaseData( FieldB::getName( ) );
        }

    private:

        //! Mapping for kernels
        MappingDesc cellDescription;

    };

} // namespace stage
} // namespace simulation
} // namespace picongpu

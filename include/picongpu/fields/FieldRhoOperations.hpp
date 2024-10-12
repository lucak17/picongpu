/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Pawel Ordyna
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/fields/FieldRho.hpp"
#include "picongpu/fields/FieldTmp.kernel"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
//#include "picongpu/param/fileOutput.param"
//#include "picongpu/particles/filter/filter.hpp"
//#include "picongpu/particles/traits/GetInterpolation.hpp"
//#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/tasks/FieldFactory.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <memory>
#include <string>
#include <iostream>


namespace picongpu
{
/** Modify this field by an another field
     *
     * @tparam AREA area where the values are modified
     * @tparam T_ModifyingOperation a binary operation defining the result of the modification as a function
     *  of two values. The 1st value is this field and the 2nd value is the modifying field.
     * @tparam T_ModifyingField type of the second field
     *
     * @param modifyingField the second field
     */
    template<uint32_t AREA, typename T_ModifyingOperation, typename T_ModifyingField, typename T_CellDesc>
    inline void modifyFieldRhoByField(
        FieldRho& fieldRho,
        T_CellDesc const& cellDescription,
        T_ModifyingField& modifyingField)
    {
        auto mapper = makeAreaMapper<AREA>(cellDescription);

        auto fieldRhoBox = fieldRho.getDeviceDataBox();
        const auto modifyingBox = modifyingField.getGridBuffer().getDeviceBuffer().getDataBox();

        using Kernel = ModifyByFieldKernel<T_ModifyingOperation, MappingDesc::SuperCellSize>;
        PMACC_LOCKSTEP_KERNEL(Kernel{}).config(
            mapper.getGridDim(),
            SuperCellSize{})(mapper, fieldRhoBox, modifyingBox);
    }
}  // namespace picongpu
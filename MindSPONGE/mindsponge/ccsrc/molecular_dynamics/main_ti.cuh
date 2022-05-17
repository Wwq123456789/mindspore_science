﻿#ifndef MAIN_CUH
#define MAIN_CUH

#include "Lennard_Jones_force/LJ_soft_core.cuh"
#include "PME_force/PME_force.cuh"
#include "TI_core/TI_core.cuh"
#include "angle/angle.cuh"
#include "bond/bond.cuh"
#include "bond/bond_soft.cuh"
#include "common.cuh"
#include "control.cuh"
#include "dihedral/dihedral.cuh"
#include "nb14/nb14.cuh"
#include "neighbor_list/neighbor_list.cuh"

void Main_Initial(int argc, char *argv[]);
void Main_Iteration();
void Main_Print();
void Main_Calculation();
void Main_Volume_Update();
void Main_Clear();

#endif
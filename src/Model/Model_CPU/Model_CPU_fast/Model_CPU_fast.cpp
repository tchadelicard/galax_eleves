#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
	std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

	const b_type b10 = b_type(10.0f);
	const b_type b1 = b_type(1.0f);

	#pragma omp parallel for
	for (int i = 0; i < n_particles; i += b_type::size)
	{
		b_type xi = b_type::load_unaligned(&particles.x[i]);
		b_type yi = b_type::load_unaligned(&particles.y[i]);
		b_type zi = b_type::load_unaligned(&particles.z[i]);

		b_type acc_x = b_type(0.0f);
		b_type acc_y = b_type(0.0f);
		b_type acc_z = b_type(0.0f);
		for (int j = 0; j < n_particles; j += b_type::size)
		{
			b_type xj = b_type::load_unaligned(&particles.x[j]);
			b_type yj = b_type::load_unaligned(&particles.y[j]);
			b_type zj = b_type::load_unaligned(&particles.z[j]);
			b_type mj = b_type::load_unaligned(&initstate.masses[j]);

			for (int k = 0; k < b_type::size; k++) {
				b_type diffx = xj - xi;
				b_type diffy = yj - yi;
				b_type diffz = zj - zi;

				b_type diffy2 = diffy * diffy;
				b_type dij = xs::fma(diffx, diffx, diffy2);
				dij = xs::fma(diffz, diffz, dij);
				dij = xs::max(dij, b1);

				// Compute inverse squared-root and handle singularities
				b_type inv_dij = xs::rsqrt(dij);
				inv_dij = inv_dij * inv_dij * inv_dij;

				dij = b10 * inv_dij;


				acc_x += diffx * dij * mj;
				acc_y += diffy * dij * mj;
				acc_z += diffz * dij * mj;

				xj = xs::rotate_right<1>(xj);
				yj = xs::rotate_right<1>(yj);
				zj = xs::rotate_right<1>(zj);
				mj = xs::rotate_right<1>(mj);
			}
		}
		acc_x.store_unaligned(&accelerationsx[i]);
		acc_y.store_unaligned(&accelerationsy[i]);
		acc_z.store_unaligned(&accelerationsz[i]);
	}

	#pragma omp parallel for
	for (int i = 0; i < n_particles; i += b_type::size)
	{
		b_type axi = b_type::load_unaligned(&accelerationsx[i]);
		b_type ayi = b_type::load_unaligned(&accelerationsy[i]);
		b_type azi = b_type::load_unaligned(&accelerationsz[i]);

		b_type pxi = b_type::load_unaligned(&particles.x[i]);
		b_type pyi = b_type::load_unaligned(&particles.y[i]);
		b_type pzi = b_type::load_unaligned(&particles.z[i]);

		b_type vxi = b_type::load_unaligned(&velocitiesx[i]);
		b_type vyi = b_type::load_unaligned(&velocitiesy[i]);
		b_type vzi = b_type::load_unaligned(&velocitiesz[i]);


		vxi += axi * b_type(2.0f);
		vyi += ayi * b_type(2.0f);
		vzi += azi * b_type(2.0f);

		pxi += vxi * b_type(0.1f);
		pyi += vyi * b_type(0.1f);
		pzi += vzi * b_type(0.1f);

		vxi.store_unaligned(&velocitiesx[i]);
		vyi.store_unaligned(&velocitiesy[i]);
		vzi.store_unaligned(&velocitiesz[i]);

		pxi.store_unaligned(&particles.x[i]);
		pyi.store_unaligned(&particles.y[i]);
		pzi.store_unaligned(&particles.z[i]);
	}
}

#endif // GALAX_MODEL_CPU_FAST

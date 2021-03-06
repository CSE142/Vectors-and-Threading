#pragma once
#include"CNN/canela.hpp"
#include"parameters.hpp"
#include"pin_tags.h"
#include"omp.h"

//#define DUMP_TENSOR_START(TAG, T) DUMP_START(TAG, (void *) &((T).data[0]), (void *) &((T).data[(T).element_count() - 1]), true)
//#define DUMP_TENSOR_STOP(TAG) DUMP_STOP(TAG)

// This class replaces its parent classes in the implementation of the learning
// model for this lab.  If you override functions in the baseclass by
// implementing them here, then the code here will run instead of the baseclass
// code.
//
// You should copy the functions you want to optimize into these classes, and
// confirm that the correctness tests pass.  Then, you can start modifying them
// to make them faster.
//
// The source code Canela is in /course/CSE141pp-SimpleCNN/CNN
class opt_fc_layer_t : public fc_layer_t
{
public:
	opt_fc_layer_t( tdsize in_size,
			int out_size ) : fc_layer_t(in_size, out_size) {

	}

//#define FC_ACTIVATE_IMPLEMENTATION g_param1_value
#define FC_ACTIVATE_IMPLEMENTATION 2
//#define CALC_GRADS_IMPLEMENTATION g_param1_value
#define CALC_GRADS_IMPLEMENTATION 2
//#define FC_ACTIVATE_THREAD_COUNT g_thread_count
//#define CALC_GRADS_THREAD_COUNT 8
//#define CALC_GRADS_THREAD_COUNT g_thread_count
	
#define I_TILE_SIZE 32
#define Y_TILE_SIZE 4
#define N_TILE_SIZE 64
	
	void activate( tensor_t<double>& in ) {
		
		//std::stringstream ss;
		
		//ss << g_function_name << "_I" << FC_ACTIVATE_IMPLEMENTATION << "_" << g_param2_value << "_" << g_param3_value << "_" << g_param4_value;
		omp_set_num_threads(8);
		//NEW_TRACE(ss.str().c_str());
		//START_TRACE();
		//DUMP_TENSOR_START("weights", weights);
		//DUMP_TENSOR_START("activator_input", activator_input);
		//DUMP_TENSOR_START("out", out);
		//DUMP_TENSOR_START("in", in);
		switch (FC_ACTIVATE_IMPLEMENTATION) {
		case 0:
			fc_layer_t::activate(in);
			break;
		case 1:
			activate_1(in);
			break;
		case 2:
			activate_2(in);
			break;
		default:
			fc_layer_t::activate(in);
			break;
		}
		//DUMP_STOP("weights");
		//DUMP_STOP("activator_input");
		//DUMP_STOP("out");
		//DUMP_STOP("in");

		//STOP_TRACE();
	}


	// This is just a demonstration of being able to set tiling
	// parameters from the commandline.  The loop nest ordering is
	// random.  Don't assume it's good.
	//
	// the __attribute__ syntax is some gcc magic that let's you
	// provide specific guidance to the compiler.  Passing
	// "noinlin" will prevent it from inlining this function into
	// activate() above.  This makes it easier to find this code in the assembly.
	void __attribute__ ((noinline)) activate_1( tensor_t<double>& in) {
		copy_input(in);
	
		tdsize old_size = in.size;
		tdsize old_out_size = out.size;
	
		// cast to correct shape
		in.size.x = old_size.x * old_size.y * old_size.z;
		in.size.y = old_size.b;
		in.size.z = 1;
		in.size.b = 1;
	
		out.size.x = old_out_size.x * old_out_size.y * old_out_size.z;
		out.size.y = old_out_size.b;
		out.size.z = 1;
		out.size.b = 1;
	
		for ( int b = 0; b < activator_input.size.b; b += 1) {
			for ( int n = 0; n < activator_input.size.x; n++ ) {
				activator_input(n, 0, 0, b) = 0;
			}
		}

//#define I_TILE_SIZE g_param2_value
//#define Y_TILE_SIZE g_param3_value
//#define N_TILE_SIZE g_param4_value
//#define I_TILE_SIZE 32
//#define Y_TILE_SIZE 4
//#define N_TILE_SIZE 16

		for ( int nn = 0; nn < out.size.x; nn+=N_TILE_SIZE ) {
			for ( int ii = 0; ii < in.size.x; ii += I_TILE_SIZE) {
				for ( int bb = 0; bb < in.size.y; bb+=Y_TILE_SIZE ) {
					for ( int b = bb; b < bb + Y_TILE_SIZE && b < in.size.y; b++ ) {
						for (int n = nn; n < nn + N_TILE_SIZE && n < out.size.x; n++ ) {
							for ( int i = ii; i < ii + I_TILE_SIZE && i < in.size.x; i++ ) {
								double in_val = in(i, b, 0);
								double weight_val = weights( i, n, 0 );
								double mul_val = in_val * weight_val;
								double acc_val = activator_input(n, 0, 0, b) + mul_val;
								activator_input(n, 0, 0, b) = acc_val;
							}
						}
					}
				}
			}
		}
	
		// finally, apply the activator function.
		for ( unsigned int n = 0; n < activator_input.element_count(); n++ ) {
			out.data[n] = activator_function( activator_input.data[n] );
		}
	
	
		in.size = old_size;
		out.size = old_out_size;
	}
	
	void __attribute__ ((noinline)) activate_2( tensor_t<double>& in) {
		copy_input(in);
	
		tdsize old_size = in.size;
		tdsize old_out_size = out.size;
	
		// cast to correct shape
		in.size.x = old_size.x * old_size.y * old_size.z;
		in.size.y = old_size.b;
		in.size.z = 1;
		in.size.b = 1;
	
		out.size.x = old_out_size.x * old_out_size.y * old_out_size.z;
		out.size.y = old_out_size.b;
		out.size.z = 1;
		out.size.b = 1;
	
		for ( int b = 0; b < activator_input.size.b; b += 1) {
			for ( int n = 0; n < activator_input.size.x; n++ ) {
				activator_input(n, 0, 0, b) = 0;
			}
		}


#pragma omp parallel for 
		for ( int nn = 0; nn < out.size.x; nn+=N_TILE_SIZE ) {
			tensor_t<double>_activator_input(activator_input.size);
			_activator_input.clear();
			for ( int ii = 0; ii < in.size.x; ii += I_TILE_SIZE) {
				for ( int bb = 0; bb < in.size.y; bb+=Y_TILE_SIZE ) {
					for ( int b = bb; b < bb + Y_TILE_SIZE && b < in.size.y; b++ ) {
						for (int n = nn; n < nn + N_TILE_SIZE && n < out.size.x; n++ ) {
							for ( int i = ii; i < ii + I_TILE_SIZE && i < in.size.x; i++ ) {
								double in_val = in(i, b, 0);
								double weight_val = weights( i, n, 0 );
								double mul_val = in_val * weight_val;
								double acc_val = activator_input(n, 0, 0, b) + mul_val;
								_activator_input(n, 0, 0, b) += acc_val;
							}
						}
					}
				}
			}
#pragma omp critical
		{
			for (int nn = 0; nn < out.size.x; nn += N_TILE_SIZE){
				for ( int bb = 0; bb < in.size.y; bb+=Y_TILE_SIZE){

					for (int b = bb; b < bb + Y_TILE_SIZE && b < in.size.y; b++){
						for (int n = nn; n < nn + N_TILE_SIZE && n < out.size.x; n++){
							activator_input(n, 0, 0, b) += _activator_input(n, 0, 0, b);
						}
					}
				}
			}
		}
		}
	
		// finally, apply the activator function.
		for ( unsigned int n = 0; n < activator_input.element_count(); n++ ) {
			out.data[n] = activator_function( activator_input.data[n] );
		}
	
	
		in.size = old_size;
		out.size = old_out_size;
	}

	void calc_grads( const tensor_t<double>& grad_next_layer ) {
		//std::stringstream ss;

		//ss << g_function_name << "_I" << CALC_GRADS_IMPLEMENTATION;
		
		omp_set_num_threads(4);

		//NEW_TRACE(ss.str().c_str());
		//START_TRACE();
		//DUMP_TENSOR_START("grads_out", grads_out);
		//DUMP_TENSOR_START("weights", weights);
		//DUMP_TENSOR_START("act_grad", act_grad);
		//DUMP_TENSOR_START("out", out);
		//DUMP_TENSOR_START("in", in);
		
		switch (CALC_GRADS_IMPLEMENTATION) {
			case 1:
				calc_grads_thread_baseline(grad_next_layer);
				break;
			case 2:
				calc_grads_thread_baseline_nn(grad_next_layer);
				break;
			case 3:
				calc_grads_thread_baseline_b(grad_next_layer);
				break;
			case 4:
				calc_grads_thread_baseline_n(grad_next_layer);
				break;
			case 5:
				calc_grads_thread_baseline_i(grad_next_layer);
				break;
			default:
				calc_grads_thread_baseline(grad_next_layer);
				break;
		}
		//DUMP_STOP("in");
		//DUMP_STOP("out");
		//DUMP_STOP("act_grad");
		//DUMP_STOP("weights");
		//DUMP_STOP("grads_out");
		//STOP_TRACE();
	}
			
	// This is as a starting point for your work on this lab.
#define BLOCK_SIZE 4	
	void calc_grads_thread_baseline( const tensor_t<double>& grad_next_layer ) {
		
		memset( grads_out.data, 0, grads_out.size.x * grads_out.size.y * grads_out.size.z * sizeof( double ) );
		
                grads_out.size.x = grads_out.size.x * grads_out.size.y * grads_out.size.z;
                grads_out.size.y = 1;
                grads_out.size.z = 1;

                for ( int b = 0; b < out.size.b; b++ ) {
                        for ( int n = 0; n < activator_input.size.x; n++ ){
				double ad = activator_derivative( activator_input(n, 0, 0, b) );
				double ng = grad_next_layer(n, 0, 0, b);
				act_grad(n, 0, 0, b) = ad * ng;
                        }
                }
		
		// Reorder loops and  tile on n
		for ( int nn = 0; nn < out.size.x; nn+=BLOCK_SIZE ) {
			for ( int b = 0; b < out.size.b; b++ ) {
				for ( int n = nn; n < nn + BLOCK_SIZE && n < out.size.x; n++ ) {
					for ( int i = 0; i < grads_out.size.x; i++ ) {
						grads_out(i, 0, 0, b) += act_grad(n, 0, 0, b) * weights( i, n, 0);
					}
				}
                        }
                }
		grads_out.size = in.size;
	}
	
	void calc_grads_thread_baseline_nn( const tensor_t<double>& grad_next_layer ) {
		
		memset( grads_out.data, 0, grads_out.size.x * grads_out.size.y * grads_out.size.z * sizeof( double ) );
		
                grads_out.size.x = grads_out.size.x * grads_out.size.y * grads_out.size.z;
                grads_out.size.y = 1;
                grads_out.size.z = 1;

                for ( int b = 0; b < out.size.b; b++ ) {
                        for ( int n = 0; n < activator_input.size.x; n++ ){
				double ad = activator_derivative( activator_input(n, 0, 0, b) );
				double ng = grad_next_layer(n, 0, 0, b);
				act_grad(n, 0, 0, b) = ad * ng;
                        }
                }
		
		// Reorder loops and  tile on n
#pragma omp parallel for 
		for ( int nn = 0; nn < out.size.x; nn+=BLOCK_SIZE ) {
			tensor_t<double>_grads_out(grads_out.size);
			_grads_out.clear();

			for ( int b = 0; b < out.size.b; b++ ) {
				for ( int n = nn; n < nn + BLOCK_SIZE && n < out.size.x; n++ ) {
					for ( int i = 0; i < grads_out.size.x; i++ ) {
						double t = act_grad(n, 0, 0, b) * weights( i, n, 0);
						_grads_out(i, 0, 0, b) += t;
					}
				}
                        }
#pragma omp critical
		{
			for (int b = 0; b < out.size.b; b++) {
				for (int i = 0; i < grads_out.size.x; i++){
					grads_out(i, 0, 0, b) += _grads_out(i, 0, 0, b);
				}
			}
                }
		}
		grads_out.size = in.size;
	}
	
	void calc_grads_thread_baseline_b( const tensor_t<double>& grad_next_layer ) {
		
		memset( grads_out.data, 0, grads_out.size.x * grads_out.size.y * grads_out.size.z * sizeof( double ) );
		
                grads_out.size.x = grads_out.size.x * grads_out.size.y * grads_out.size.z;
                grads_out.size.y = 1;
                grads_out.size.z = 1;

                for ( int b = 0; b < out.size.b; b++ ) {
                        for ( int n = 0; n < activator_input.size.x; n++ ){
				double ad = activator_derivative( activator_input(n, 0, 0, b) );
				double ng = grad_next_layer(n, 0, 0, b);
				act_grad(n, 0, 0, b) = ad * ng;
                        }
                }
		
		// Reorder loops and  tile on n
		for ( int nn = 0; nn < out.size.x; nn+=BLOCK_SIZE ) {
#pragma omp parallel for
			for ( int b = 0; b < out.size.b; b++ ) {
				for ( int n = nn; n < nn + BLOCK_SIZE && n < out.size.x; n++ ) {
					for ( int i = 0; i < grads_out.size.x; i++ ) {
						grads_out(i, 0, 0, b) += act_grad(n, 0, 0, b) * weights( i, n, 0);
					}
				}
                        }
                }
		grads_out.size = in.size;
	}
	
	void calc_grads_thread_baseline_n( const tensor_t<double>& grad_next_layer ) {
		
		memset( grads_out.data, 0, grads_out.size.x * grads_out.size.y * grads_out.size.z * sizeof( double ) );
		
                grads_out.size.x = grads_out.size.x * grads_out.size.y * grads_out.size.z;
                grads_out.size.y = 1;
                grads_out.size.z = 1;

                for ( int b = 0; b < out.size.b; b++ ) {
                        for ( int n = 0; n < activator_input.size.x; n++ ){
				double ad = activator_derivative( activator_input(n, 0, 0, b) );
				double ng = grad_next_layer(n, 0, 0, b);
				act_grad(n, 0, 0, b) = ad * ng;
                        }
                }
		
		// Reorder loops and  tile on n
		for ( int nn = 0; nn < out.size.x; nn+=BLOCK_SIZE ) {
			for ( int b = 0; b < out.size.b; b++ ) { 
				int minn = std::min(nn + BLOCK_SIZE, out.size.x);
#pragma omp parallel for
				for ( int n = nn; n < minn; n++ ) {
					tensor_t<double>_grads_out(grads_out.size);
					_grads_out.clear();
					for ( int i = 0; i < grads_out.size.x; i++ ) {
						double t = act_grad(n, 0, 0, b) * weights( i, n, 0);
						_grads_out(i, 0, 0, b) += t;
					}
#pragma omp critical
				{
					for (int i = 0; i < grads_out.size.x; i++) {
						grads_out(i, 0, 0, b) += _grads_out(i, 0, 0, b);
					}
				}
				}
                        }
                }
		grads_out.size = in.size;
	}
	
	void calc_grads_thread_baseline_i( const tensor_t<double>& grad_next_layer ) {
		
		memset( grads_out.data, 0, grads_out.size.x * grads_out.size.y * grads_out.size.z * sizeof( double ) );
		
                grads_out.size.x = grads_out.size.x * grads_out.size.y * grads_out.size.z;
                grads_out.size.y = 1;
                grads_out.size.z = 1;

                for ( int b = 0; b < out.size.b; b++ ) {
                        for ( int n = 0; n < activator_input.size.x; n++ ){
				double ad = activator_derivative( activator_input(n, 0, 0, b) );
				double ng = grad_next_layer(n, 0, 0, b);
				act_grad(n, 0, 0, b) = ad * ng;
                        }
                }
		
		// Reorder loops and  tile on n
		for ( int nn = 0; nn < out.size.x; nn+=BLOCK_SIZE ) {
			for ( int b = 0; b < out.size.b; b++ ) {
				for ( int n = nn; n < nn + BLOCK_SIZE && n < out.size.x; n++ ) {
#pragma omp parallel for
					for ( int i = 0; i < grads_out.size.x; i++ ) {
						grads_out(i, 0, 0, b) += act_grad(n, 0, 0, b) * weights( i, n, 0);
					}
				}
                        }
                }
		grads_out.size = in.size;
	}
			
};
	

class opt_conv_layer_t: public conv_layer_t
{
public:
	
	opt_conv_layer_t( uint16_t stride,
			  uint16_t kernel_size, 
			  uint16_t kernel_count,
			  double pad,
			  tdsize in_size
			  ) : conv_layer_t(stride, kernel_size, kernel_count, pad, in_size){}

	void calc_grads(const tensor_t<double>& grad_next_layer ) {

		throw_assert(grad_next_layer.size == out.size, "mismatch input size for calc_grads");
		omp_set_num_threads(4);
		for ( int b = 0; b < in.size.b; b++ )
	for ( uint k = 0; k < filter_grads.size(); k++ ) 
		for ( int z = 0; z < in.size.z; z++ )			
			for ( int j = 0; j < kernel_size; j++ )				
					for ( int i = 0; i < kernel_size; i++ )	
						
							filter_grads[k].get( i, j, z, b ).grad = 0;
#pragma omp parallel for simd
	for ( int b = 0; b < in.size.b; b++ ) {
		for ( int z = 0; z < in.size.z; z++ ) {
			for ( int y = 0; y < in.size.y; y++ ) {
				for ( int x = 0; x < in.size.x; x++ ) {
					range_t rn = map_to_output( x, y );					
						double sum_error = 0;

						for ( int i = rn.min_x; i <= rn.max_x; i++ ) {
							int minx = i * stride;
							for ( int j = rn.min_y; j <= rn.max_y; j++ ) {
								int miny = j * stride;
								for ( int k = rn.min_z; k <= rn.max_z; k++ ) {
									int w_applied = filters[k].get( x - minx, y - miny, z );
									sum_error += w_applied * grad_next_layer( i, j, k, b );
									filter_grads[k].get( x - minx, y - miny, z, b ).grad += in( x, y, z, b ) * grad_next_layer( i, j, k, b );
								}
							}
						}
						grads_out( x, y, z, b ) = sum_error;
					}
				}
			}
		}
	}

        void fix_weights() {
                for ( int b = 0; b < in.size.b; b++ )
                        for ( uint a = 0; a < filters.size(); a++ )
                                for ( int i = 0; i < kernel_size; i++ )
                                        for ( int j = 0; j < kernel_size; j++ )
					
                                                for ( int z = 0; z < in.size.z; z++ ) {
                                                        double& w = filters[a].get( i, j, z );
                                                        gradient_t& grad = filter_grads[a].get( i, j, z, b );
                                                        w = update_weight( w, grad );
                                                        update_gradient( grad );
                                                }
        }
	

	void activate( tensor_t<double>& in ) {
		omp_set_num_threads(4);
                copy_input(in);
		
#pragma omp parallel for simd

                for ( int b = 0; b < out.size.b; b++ ) {
                        for ( uint filter = 0; filter < filters.size(); filter++ ) {
                                tensor_t<double>& filter_data = filters[filter];
				for ( int y = 0; y < out.size.y; y++ ) {
                                	for ( int x = 0; x < out.size.x; x++ ) {
                                        
                                                point_t mapped(x*stride, y*stride, 0);
                                                double sum = 0;
						
                                                for ( int i = 0; i < kernel_size; i++ )
                                                        for ( int j = 0; j < kernel_size; j++ )
								for ( int z = 0; z < in.size.z; z++ )
                                                                 {
                                                                        double f = filter_data( i, j, z );

                                                                        double v;
                                                                        if (mapped.x + i >= in.size.x ||
                                                                        mapped.y + j >= in.size.y) {
                                                                                v = pad;
                                                                        } else {
                                                                                v = in( mapped.x + i, mapped.y + j, z, b );
                                                                        }
                                                                        sum += f*v;
                                                                }
                                                out( x, y, filter, b ) = sum;
                                        }
                                }
                        }
                }
        }
};

class opt_pool_layer_t: public pool_layer_t
{
public:
	opt_pool_layer_t( uint16_t stride,
			  uint16_t filter_size,
			  double pad,
			  tdsize in_size ) : pool_layer_t(stride, filter_size, pad, in_size){}
			  
	void calc_grads(const tensor_t<double>& grad_next_layer )
        {
	omp_set_num_threads(4);
#pragma omp parallel for simd

                for ( int b = 0; b < in.size.b; b++ ) {
                        
                           for ( int y = 0; y < in.size.y; y++ ) {
				for ( int x = 0; x < in.size.x; x++ ) {
                                        range_t rn = map_to_output( x, y );
                                        for ( int z = 0; z < in.size.z; z++ ) {
                                                double sum_error = 0;
                                                for ( int i = rn.min_x; i <= rn.max_x; i++ ) {
                                                        for ( int j = rn.min_y; j <= rn.max_y; j++ ) {
							
                                                                int is_max = in( x, y, z ) == out( i, j, z ) ? 1 : 0;
                                                                sum_error += is_max * grad_next_layer( i, j, z );
                                                        }
                                                }
                                                grads_out( x, y, z, b ) = sum_error;
                                        }
                                }
                        }
                }
        }
	
	void activate(tensor_t<double>& in ) {
                copy_input(in);
		
		omp_set_num_threads(4);
#pragma omp parallel for simd

                for ( int b = 0; b < out.size.b; b++ ) {
                        for ( int x = 0; x < out.size.x; x++ ) {
                                for ( int y = 0; y < out.size.y; y++ ) {
                                        for ( int z = 0; z < out.size.z; z++ ) {
                                                point_t mapped(x*stride, y*stride, 0);
                                                double mval = -FLT_MAX;
                                                for ( int i = 0; i < filter_size; i++ )
                                                        for ( int j = 0; j < filter_size; j++ ) {
                                                                double v;
                                                                if (mapped.x + i >= in.size.x ||
                                                                mapped.y + j >= in.size.y) {
                                                                        v = pad;
                                                                } else {
                                                                        v = in( mapped.x + i, mapped.y + j, z );
                                                                }

                                                                if ( v > mval )
                                                                        mval = v;
                                                        }
                                                out( x, y, z, b ) = mval;
                                        }
                                }
                        }
                }
        }
};

class opt_relu_layer_t : public relu_layer_t
{
public:
	opt_relu_layer_t(const tdsize & in_size )
		:
		relu_layer_t(in_size)
	{
	}
	
};

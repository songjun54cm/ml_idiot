THEANO_FLAGS=mode=FAST_COMPILE,exception_verbosity=high,optimizer=None,optimizer_excluding=inplace,reoptimize_unpickled_function=False,floatX=float32,device=gpu2,allow_gc=False,scan.allow_gc=False,nvcc.flags=-use_fast_math python $*
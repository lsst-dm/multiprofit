import multiprofit.tests as mpftest
test = mpftest.gaussian_test(nbenchmark=200, do_grad=True, do_jac=True, nsub=4) 
for x in test: 
    print('re={} q={:.2f} ang={:2.0f} {}'.format(x['reff'], x['axrat'], x['ang'], x['string']))


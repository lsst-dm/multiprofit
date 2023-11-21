from test_utils import gaussian_test
from timeit import default_timer as timer

start = timer()
test = gaussian_test(
    nbenchmark=100, do_like=True, do_residual=True, do_grad=True, do_jac=True,
    do_meas_modelfit=False, nsub=4,
)
for x in test:
    print(f"re={x['reff']:.3f} q={x['axrat']:.2f}"
          f" ang={x['ang']:2.1f} { x['string']}")
print(f'Test complete in {timer() - start:.2f}s')

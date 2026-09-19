is_simple_core = True

if is_simple_core:
    from liouville.core_simple import Variable
    from liouville.core_simple import Function
    from liouville.core_simple import using_config
    from liouville.core_simple import no_grad
    from liouville.core_simple import as_array
    from liouville.core_simple import as_variable
    from liouville.core_simple import setup_variable

setup_variable()
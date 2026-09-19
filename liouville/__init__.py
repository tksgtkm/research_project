is_simple_core = False

if is_simple_core:
    from liouville.core_simple import Variable
    from liouville.core_simple import Function
    from liouville.core_simple import using_config
    from liouville.core_simple import no_grad
    from liouville.core_simple import as_array
    from liouville.core_simple import as_variable
    from liouville.core_simple import setup_variable
else:
    from liouville.core import Variable
    from liouville.core import Function
    from liouville.core import using_config
    from liouville.core import no_grad
    from liouville.core import as_array
    from liouville.core import as_variable
    from liouville.core import setup_variable
    from liouville.core import Config

    import liouville.functions
    import liouville.utils

setup_variable()
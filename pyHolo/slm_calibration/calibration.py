
from . import get_raw_response
from . import get_holo_response


def calibrate_main(**kwargs):

    if not kwargs.get('use_raw_pkl', False):
        raw_data = get_raw_response(**kwargs)
        kwargs.update(raw_response=raw_data)

    if not kwargs.get('only_raw', False):
        get_holo_response(**kwargs)
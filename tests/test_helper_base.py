from hoocs import helper_base


def test_organize_key():
    key_new = helper_base.organize_key(key='empty_set')
    assert key_new == 'empty_set', 'Alters incorrect keys. '

    key_new = helper_base.organize_key(key='10')
    assert key_new == '10', 'Keys are not properly sorted.'

    key_new = helper_base.organize_key(key='1^10^99^111')
    assert key_new == '1^10^99^111', 'Keys are not properly sorted.'

    key_new = helper_base.organize_key(key='2^4^1^3')
    assert key_new == '1^2^3^4', 'Keys are not properly sorted.'

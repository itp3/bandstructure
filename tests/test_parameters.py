import pytest
from bandstructure import Parameters


def test_simple():
    p = Parameters({'a': 1, 'b': "foo"})
    assert p['a'] == 1
    assert p['b'] == "foo"

    p['a'] = "bar"

    assert p['a'] == "bar"


def test_default():
    p = Parameters({'a': 1, 'b': "foo"})

    assert p.get('c', "default") == "default"

    with pytest.raises(KeyError):
        p.get('c')

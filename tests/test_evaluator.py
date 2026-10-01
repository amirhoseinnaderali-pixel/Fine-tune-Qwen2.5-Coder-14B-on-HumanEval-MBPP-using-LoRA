from scripts.evaluate_mbpp import extract_code


def test_extract_code():
    fence = chr(96) * 3
    source = fence + "python\nprint(1)\n" + fence
    assert extract_code(source) == "print(1)"

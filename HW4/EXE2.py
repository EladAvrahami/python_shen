def namecheck(str1, str2):
    result = []
    for ch in str1:
        if ch not in str2:
            result.append(ch)
    for ch in str2:
        if ch not in str1:
            result.append(ch)
    return result

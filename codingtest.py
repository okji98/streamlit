my_string = "He11oWor1d"
overwrite_string = "lloWorl"
s = 2

# def solution(my_string, overwrite_string, s):
#     answer = ''
#     return answer
print(my_string[s:])
print(len(overwrite_string))
print(my_string[s:len(overwrite_string) + 2])
# ov_len = len(overwrite_string)
my_string = my_string.replace(my_string[s:len(overwrite_string) + 2], overwrite_string)
print(my_string)
# while True:
#     if len(overwrite_string)
#     my_string[s:] = overwrite_string
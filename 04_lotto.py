import streamlit as st
import random
import datetime
# 로또번호는 1~45 랜덤하게 가져오되 중복체크가 안되니 set에다가 넣기
st.header(':sparkles:로또 생성기:sparkles:')

button = st.button('로또 생성 버튼')

def generate_lotto():
    lotto_sets = []
    for _ in range(5):
        numbers = set()
        while len(numbers) < 6:
            numbers.add(random.randint(1, 45))
        lotto_sets.append(sorted(numbers))
    return lotto_sets

if button:
    lotto_numbers = generate_lotto()
    for idx, numbers in enumerate(lotto_numbers, 1):
        st.write(f"게임 {idx}: {numbers}")
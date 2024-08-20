def calculate_bmr(weight, height, age, gender):
    if gender == '남':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    elif gender == '여':
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    else:
        raise ValueError("Invalid gender. Please input '남' or '여'.")
    return bmr

def calculate_tdee(bmr, activity_level):
    activity_factors = {
        1: 1.2,    # 매우 적음
        2: 1.375,  # 가벼운 활동
        3: 1.55,   # 보통 활동
        4: 1.725,  # 활동적
        5: 1.9     # 매우 활동적
    }

    if activity_level not in activity_factors:
        raise ValueError("Invalid activity level. Please input a number between 1 and 5.")

    tdee = bmr * activity_factors[activity_level]
    return tdee

def calculate_calorie_deficit(weight_loss_kg):
    # 체지방 1kg 감량에 필요한 칼로리
    calories_per_kg_fat = 7700
    return weight_loss_kg * calories_per_kg_fat

def calculate_diet_duration(calorie_deficit, tdee):
    daily_calorie_deficit = tdee * 0.15
    days_required = calorie_deficit / daily_calorie_deficit
    return days_required

def return_message(user_info):
    print("다이어트 예상 기간 계산기")

    gender = input("성별을 입력하세요 (남/여): ").strip()
    weight = float(input("현재 몸무게를 입력하세요 (kg): ").strip())
    height = float(input("키를 입력하세요 (cm): ").strip())
    age = int(input("나이를 입력하세요: ").strip())

    print("\n활동 정도를 선택하세요:")
    print("1: 매우 적음 (거의 운동하지 않음)")
    print("2: 가벼운 활동 (주 1-3회 가벼운 운동)")
    print("3: 보통 활동 (주 3-5회 보통 운동)")
    print("4: 활동적 (주 6-7회 강도 높은 운동)")
    print("5: 매우 활동적 (매우 강도 높은 운동, 육체 노동)")

    activity_level = int(input("활동 정도를 입력하세요 (1-5): ").strip())
    target_weight = float(input("목표 체중을 입력하세요 (kg): ").strip())

    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity_level)

    weight_loss = weight - target_weight
    calorie_deficit = calculate_calorie_deficit(weight_loss)

    diet_duration_days = calculate_diet_duration(calorie_deficit, tdee)

    print(f"\n당신의 기초대사량은 {bmr:.2f} kcal/day 입니다.")
    print(f"당신의 활동대사량은 {tdee:.2f} kcal/day 입니다.")
    print(f"감량해야 하는 칼로리는 총 {calorie_deficit:.2f} kcal 입니다.")
    print(f"예상 다이어트 기간은 약 {diet_duration_days:.2f} 일입니다.")

if __name__ == "__main__":
    main()

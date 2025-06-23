import math
import statistics  # ספרייה שימושית לחישובים סטטיסטיים


def calculate_t_confidence_interval():
    """
    Calculates the confidence interval for the mean when the population
    standard deviation is unknown (t-distribution).
    """
    print("\n--- תכנית ב': חישוב רווח סמך עם סטיית תקן לא ידועה (T) ---")

    #  : מאגר נתונים רזה המדמה טבלת T
    t_scores = {
        # df: {90: t_val, 95: t_val, 99: t_val}
        7: {90: 1.895, 95: 2.365, 99: 3.499},  # for n=8
        8: {90: 1.860, 95: 2.306, 99: 3.355},  # for n=9
        9: {90: 1.833, 95: 2.262, 99: 3.250},  # for n=10
        10: {90: 1.812, 95: 2.228, 99: 3.169},  # for n=11
        11: {90: 1.796, 95: 2.201, 99: 3.106}  # for n=12
    }

    #  : קליטת נתונים מהמשתמש עם בדיקת תקינות
    try:
        #  : קליטת גודל המדגם ובדיקה שהוא בין 8 ל-12
        while True:
            n = int(input(f"הכנס את כמות הציונים במדגם (n, בין 8 ל-12): "))  #
            if 8 <= n <= 12:
                break
            print("שגיאה: כמות הציונים חייבת להיות בין 8 ל-12.")

        #  : קליטת הציונים עצמם מהמשתמש
        scores = []
        print(f"הכנס את {n} הציונים שהתקבלו במדגם (הקש Enter אחרי כל ציון):")
        for i in range(n):
            score = float(input(f"ציון מספר {i + 1}: "))
            scores.append(score)

        #  : קליטת רמת הביטחון
        while True:
            confidence_level = int(input("הכנס את רמת הביטחון הרצויה באחוזים (90, 95, או 99): "))
            if confidence_level in [90, 95, 99]:
                break
            print("שגיאה: אנא בחר רמת בטחון של 90, 95 או 99.")

    except ValueError:
        print("שגיאה: אחד או יותר מהערכים שהוזנו אינו תקין.")
        return

    # --- החישובים ---
    #  : חישוב ממוצע וסטיית תקן של המדגם
    sample_mean = statistics.mean(scores)
    sample_std = statistics.stdev(scores)

    #  חישוב דרגות חופש
    degrees_of_freedom = n - 1

    #  בחירת ערך ה-T המתאים מהמאגר
    t_value = t_scores[degrees_of_freedom][confidence_level]

    # : חישוב שגיאת התקן
    standard_error = sample_std / math.sqrt(n)

    # : חישוב מרווח הטעות
    margin_of_error = t_value * standard_error

    # : חישוב גבולות רווח הסמך
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    #  : הצגת התוצאה
    print("\n" + "=" * 25)
    print("      תוצאות החישוב")
    print("=" * 25)
    print(f"ממוצע המדגם (x̄) שחושב הוא: {sample_mean:.4f}")
    print(f"סטיית התקן של המדגם (s) שחושבה היא: {sample_std:.4f}")
    print(f"ברמת ביטחון של {confidence_level}%, רווח הסמך לתוחלת האוכלוסייה (μ) הוא:")
    print(f"גבול עליון וגבול תחתון ")
    print(f"[{lower_bound:.4f}, {upper_bound:.4f}]")
    print("=" * 25)



calculate_t_confidence_interval()
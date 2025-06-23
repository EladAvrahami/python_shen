import math


def get_z_score_from_table(confidence_percent):
    """
    This function simulates looking up a Z-score from a cumulative table
    based on a given confidence level. The values here are pre-calculated
    from the standard normal table for common confidence levels.
    """
    # המאגר מבוסס על חיפוש בטבלת Z כפי שהוסבר
    # לדוגמה, עבור 95%, חיפשנו בטבלה את הערך 0.975
    z_scores_from_table = {
        90: 1.645,
        95: 1.960,
        99: 2.576
    }
    return z_scores_from_table.get(confidence_percent)


def calculate_z_confidence_interval():
    """
    Calculates the confidence interval for the mean when the population
    standard deviation is known (Z-distribution).
    """
    print("--- תכנית א' (מתוקנת): חישוב רווח סמך עם טבלת Z ---")

    try:
        sample_mean = float(input("הכנס את ממוצע הציונים של המדגם (x̄): "))

        while True:
            n = int(input(f"הכנס את כמות הציונים במדגם (n, בין 8 ל-12): "))  #
            if 8 <= n <= 12:
                break
            print("שגיאה: כמות הציונים חייבת להיות בין 8 ל-12.")

        population_std = float(input("הכנס את סטיית התקן של האוכלוסייה (σ): "))

        while True:
            confidence_level = int(input("הכנס את רמת הביטחון הרצויה באחוזים (90, 95, או 99): "))
            z_value = get_z_score_from_table(confidence_level)
            if z_value is not None:
                break
            print("שגיאה: אנא בחר רמת בטחון נתמכת (90, 95, או 99).")

    except ValueError:
        print("שגיאה: קלט לא תקין. אנא הזן מספרים בלבד.")
        return

    # --- חישובים ---
    # חישוב שגיאת התקן
    standard_error = population_std / math.sqrt(n)

    # חישוב מרווח הטעות
    margin_of_error = z_value * standard_error

    # חישוב גבולות רווח הסמך
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    # --- הצגת התוצאה ---
    print("\n" + "=" * 20)
    print("      תוצאות החישוב")
    print("=" * 20)
    print(f"נתונים שהוזנו:")
    print(f"  - ממוצע מדגם (x̄): {sample_mean}")
    print(f"  - גודל מדגם (n): {n}")
    print(f"  - סטיית תקן אוכלוסייה (σ): {population_std}")
    print(f"  - רמת ביטחון: {confidence_level}% (ערך Z={z_value})")
    print("\nרווח הסמך לתוחלת האוכלוסייה (μ) הוא:")
    print(f"[{lower_bound:.4f}, {upper_bound:.4f}]")
    print("=" * 20)


# הרצת התכנית
calculate_z_confidence_interval()
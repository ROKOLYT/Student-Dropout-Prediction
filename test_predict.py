from fastapi.testclient import TestClient
from predict import app

client = TestClient(app)

# Sample 1: High risk student (older, fees unpaid, debtor, 0 units passed in 2nd semester)
student_high_risk = {
    'marital_status': 1,
    'application_mode': 17,
    'application_order': 2,
    'course': 9254,
    'daytime_evening_attendance': 1,
    'previous_qualification': 1,
    'previous_qualification_grade': 115.0,
    'nacionality': 1,
    'mothers_qualification': 19,
    'fathers_qualification': 19,
    'mothers_occupation': 5,
    'fathers_occupation': 9,
    'admission_grade': 110.0,
    'displaced': 0,
    'educational_special_needs': 0,
    'debtor': 1,
    'tuition_fees_up_to_date': 0,
    'gender': 1,
    'scholarship_holder': 0,
    'age_at_enrollment': 32,
    'international': 0,
    'curricular_units_1st_sem_credited': 0,
    'curricular_units_1st_sem_enrolled': 6,
    'curricular_units_1st_sem_evaluations': 4,
    'curricular_units_1st_sem_approved': 1,
    'curricular_units_1st_sem_grade': 10.0,
    'curricular_units_1st_sem_without_evaluations': 0,
    'curricular_units_2nd_sem_credited': 0,
    'curricular_units_2nd_sem_enrolled': 6,
    'curricular_units_2nd_sem_evaluations': 2,
    'curricular_units_2nd_sem_approved': 0,
    'curricular_units_2nd_sem_grade': 0.0,
    'curricular_units_2nd_sem_without_evaluations': 0,
    'unemployment_rate': 12.4,
    'inflation_rate': 1.2,
    'gdp': 0.5
}

# Sample 2: Low risk student (younger, fees paid, scholarship holder, passed all units)
student_low_risk = {
    'marital_status': 1,
    'application_mode': 1,
    'application_order': 1,
    'course': 9254,
    'daytime_evening_attendance': 1,
    'previous_qualification': 1,
    'previous_qualification_grade': 160.0,
    'nacionality': 1,
    'mothers_qualification': 1,
    'fathers_qualification': 3,
    'mothers_occupation': 3,
    'fathers_occupation': 3,
    'admission_grade': 145.0,
    'displaced': 1,
    'educational_special_needs': 0,
    'debtor': 0,
    'tuition_fees_up_to_date': 1,
    'gender': 0,
    'scholarship_holder': 1,
    'age_at_enrollment': 19,
    'international': 0,
    'curricular_units_1st_sem_credited': 0,
    'curricular_units_1st_sem_enrolled': 6,
    'curricular_units_1st_sem_evaluations': 6,
    'curricular_units_1st_sem_approved': 6,
    'curricular_units_1st_sem_grade': 14.5,
    'curricular_units_1st_sem_without_evaluations': 0,
    'curricular_units_2nd_sem_credited': 0,
    'curricular_units_2nd_sem_enrolled': 6,
    'curricular_units_2nd_sem_evaluations': 6,
    'curricular_units_2nd_sem_approved': 6,
    'curricular_units_2nd_sem_grade': 14.0,
    'curricular_units_2nd_sem_without_evaluations': 0,
    'unemployment_rate': 10.8,
    'inflation_rate': 1.4,
    'gdp': 1.74
}

def run_tests():
    print("Testing GET / (Health check):")
    res = client.get("/")
    print(f"Status: {res.status_code}, Response: {res.json()}")
    assert res.status_code == 200

    print("\nTesting POST /predict for High Risk Student:")
    res_high = client.post("/predict", json=student_high_risk)
    print(f"Status: {res_high.status_code}, Response: {res_high.json()}")
    assert res_high.status_code == 200
    assert res_high.json()["dropout_prediction"] is True

    print("\nTesting POST /predict for Low Risk Student:")
    res_low = client.post("/predict", json=student_low_risk)
    print(f"Status: {res_low.status_code}, Response: {res_low.json()}")
    assert res_low.status_code == 200
    assert res_low.json()["dropout_prediction"] is False

    print("\nAll API tests passed successfully!")

if __name__ == "__main__":
    run_tests()

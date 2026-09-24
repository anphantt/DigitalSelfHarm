from pathlib import Path
import os

ROOT = Path(os.environ.get("DSH_ROOT", "home/anphan/Documents/DigitalSelfHarm"))
DATA_PATH = Path(
    os.environ.get(
        "DSH_DATA_PATH",
        str(ROOT / "data" / "data_processed.csv"),
    )
)


OUTPUT_ROOT = Path(
    os.environ.get(
        "DSH_OUTPUT_ROOT",
        str(ROOT / "github"/ "output"),
    )
)

TARGET_NAME = "label_year"
CV_SEEDS = [42, 100, 2222]
N_SPLITS = 10
MODEL_SEED = 42

EXCLUDE_MODEL_FEATURES = [
    "missing_count",
    "MISS_COUNT",
    "MISS_RATE",
    "answered_rate",
]

IMBALANCE_STRATEGIES = [
    "PlainXGBoost",
    "WeightedXGBoost",
    "ROSXGBoost",
    "EasyEnsembleXGB",
    "BalanceCascadeXGB",
]
BALANCE_CASCADE_N_ESTIMATORS = 100

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260907

MAIN_MODELS = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "EasyEnsembleXGB",
    "BalanceCascadeXGB",
]

MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "EasyEnsembleXGB": "EasyEnsembleXGB",
    "BalanceCascadeXGB": "BalanceCascadeXGB",
}


N_TREES = 300
MAX_DEPTH = 6
LR_MAX_ITER = 5000
EASY_ENSEMBLE_N_ESTIMATORS = 100

XGB_TREE_METHOD = "hist"


ABLATION_MODELS = [
    "LogisticRegression",
    "XGBoost",
]

RISK_SCORE_DECILES = 10
RISK_STRATIFICATION_MODEL = "XGBoost"
PRIORITIZATION_CAPACITIES = [0.05, 0.10, 0.20]
# The submitted fixed-capacity table used these four models; BalanceCascadeXGB
# remains a primary performance model but is not included in that table.
PRIORITIZATION_MODELS = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "EasyEnsembleXGB",
]


SHAP_MODEL = "XGBoost"
SHAP_TOP_K = 15
SHAP_PLOT_MAX_ROWS = 5000


INCLUDE_OVERALL_COLUMN = False

DEMOGRAPHIC_CONTINUOUS = {"Age": "D1"}
DEMOGRAPHIC_CATEGORICAL = {"Grade": "D2"}

DEMOGRAPHIC_ONEHOT = {
    "Sex": "D3_",
    "Primary language": "D8_",
    "Living area": "Living_",
}

DEMOGRAPHIC_BINARY = {
    "White": "WHITE",
    "Black": "BLACK",
    "Hispanic": "HISPANIC",
    "Other race": "OTHER_RACE",
}

AGE_GRADE = ["D1", "D2"]

RACE_ETHNICITY = ["OTHER_RACE", "WHITE", "BLACK", "HISPANIC"]

HOUSEHOLD_COMPOSITION = [
    "D5A", "D5B", "D5C", "D5D", "D5E", "D5F", "D5G", "D5H",
    "D5I", "D5J", "D5K", "D5L", "D5M", "D5N", "D5O", "D5P",
]

PARENT_EDUCATION = ["D9", "D10"]

LANGUAGE_HOME = ["D8_Another Language", "D8_English", "D8_Missing", "D8_Spanish"]

SEX = ["D3_Female", "D3_Male", "D3_Missing"]

LIVING_AREA = ["Living_Missing", "Living_city", "Living_country", "Living_farm"]

DEMOGRAPHIC_BACKGROUND = sorted(set(
    AGE_GRADE
    + RACE_ETHNICITY
    + HOUSEHOLD_COMPOSITION
    + PARENT_EDUCATION
    + LANGUAGE_HOME
    + SEX
    + LIVING_AREA
))

DEPRESSION = ["FL6", "FL7", "FL8", "FL9"]
BULLYING_VICTIMIZATION = ["FL48b", "FL60X", "FL61X", "FL62X"]
BULLYING_PERPETRATION = ["FL63X", "FL64X", "FL65X"]
SCHOOL_SUPPORT_OPPORTUNITIES = ["Q14", "Q2891", "Q15", "Q2057", "Q17", "Q18", "Q21", "Q731"]
SCHOOL_ENGAGEMENT = ["Q3681", "Q3682", "Q3683", "Q3684", "Q3685", "Q3686", "Q3668"]
ACADEMIC_PERFORMANCE = ["Q13", "Q23"]
SCHOOL_TRUANCY = ["Q738"]
PEER_SUBSTANCE_USE = ["Q58A", "Q58B", "Q58C", "FL700", "FL701", "FL68"]
PEER_NORMS = ["Q59AX", "Q59BX", "Q59CX", "FL706", "FL707", "Q59DX"]
SUBSTANCE_INITIATION_AGE = ["Q60A", "Q60B", "FL702", "FL703", "Q60C", "Q60D"]

SUBSTANCE_USE_FREQUENCY = [
    "U3", "U4", "U5", "U6", "U7",
    "FL10", "FL11", "FL712", "FL713", "FL714", "FL715",
    "U10", "U11", "U16", "U17",
    "FL49", "FL50", "FL66", "FL67", "FL51", "FL52",
    "U30X", "U31X",
    "FL46", "FL47", "FL55", "FL56", "FL24", "FL25",
    "U24", "U25",
]

SCHOOL_TIME_SUBSTANCE_USE = ["FL122", "FL123", "FL124"]

PERSONAL_NORMS = [
    "Q61A", "Q61B", "Q61C", "Q61D", "Q61E",
    "Q67A", "Q67B", "Q67C", "Q67D",
    "FL704", "FL705",
]

PERCEIVED_SUBSTANCE_HARM = [
    "Q3687", "Q3679", "Q3688X", "FL710", "FL711",
    "Q3680", "FL120", "FL116",
]

PARENTAL_SUBSTANCE_NORMS_RULES = ["Q74AX", "Q74B", "Q74C", "FL125", "Q76"]
PARENTAL_SUPPORT_CONNECTION = ["Q78", "Q86", "Q89", "Q91", "Q93", "Q94", "Q96", "Q99"]
PARENTAL_MONITORING_SUPERVISION = ["Q80", "FL510X", "Q85", "Q84", "Q83", "Q82", "Q79", "FL508"]
FAMILY_CONFLICT = ["Q2910", "Q2911", "Q2909"]
FAMILY_SUBSTANCE_PROBLEMS = ["Q77"]
SUBSTANCE_WEAPON_ACCESS = ["Q25", "Q26", "Q28", "Q30", "Q32"]
PERCEIVED_ENFORCEMENT = ["Q27", "Q29"]
COMMUNITY_ADULT_NORMS = ["Q33A", "Q33B", "Q33C"]
NEIGHBORHOOD_DISORDER = ["Q103A", "Q103B", "Q103C", "Q103D"]
NEIGHBORHOOD_SAFETY = ["Q107"]

RESIDENTIAL_SCHOOL_MOBILITY = ["Q104", "Q106", "Q108", "Q110"]
EXTRACURRICULAR_ACTIVITIES = ["FL40", "FL41", "FL42", "FL43", "FL44"]
RELIGIOUS_INVOLVEMENT = ["Q54"]

STRUCTURED_ACTIVITY_INVOLVEMENT = sorted(set(
    EXTRACURRICULAR_ACTIVITIES + RELIGIOUS_INVOLVEMENT
))

IMPULSIVITY_IRRITABILITY = ["FL502", "FL503", "FL504", "FL505", "FL506", "FL507"]
EXTERNALIZING_INITIATION_AGE = ["Q60E", "Q60F", "Q60G", "Q60H"]
DELINQUENCY_VIOLENCE = ["Q66A", "Q66B", "Q66C", "Q66D", "Q66E", "Q66F", "Q66H"]
SLEEP_DURATION = ["FL509"]
DOMAIN_GROUPS = {
    "Demographic_background": DEMOGRAPHIC_BACKGROUND,
    "Internalizing": DEPRESSION,
    "Bullying": sorted(set(BULLYING_VICTIMIZATION + BULLYING_PERPETRATION)),
    "School_context": sorted(set(
        SCHOOL_SUPPORT_OPPORTUNITIES 
        + SCHOOL_ENGAGEMENT 
        + ACADEMIC_PERFORMANCE 
        + SCHOOL_TRUANCY
        )),
    "Peer_context": sorted(set(PEER_NORMS + PEER_SUBSTANCE_USE)),
    "Substance_use": sorted(set(
        SUBSTANCE_INITIATION_AGE 
        + SUBSTANCE_USE_FREQUENCY 
        + SCHOOL_TIME_SUBSTANCE_USE
        )),
    "Personal_attitudes_risk_perception": sorted(set(PERSONAL_NORMS + PERCEIVED_SUBSTANCE_HARM)),
    "Family_context": sorted(set(
        PARENTAL_SUBSTANCE_NORMS_RULES
        + PARENTAL_SUPPORT_CONNECTION
        + PARENTAL_MONITORING_SUPERVISION
        + FAMILY_CONFLICT
        + FAMILY_SUBSTANCE_PROBLEMS
    )),

    "Community_environment": sorted(set(
        SUBSTANCE_WEAPON_ACCESS
        + PERCEIVED_ENFORCEMENT
        + COMMUNITY_ADULT_NORMS
        + NEIGHBORHOOD_DISORDER
        + NEIGHBORHOOD_SAFETY
    )),

    "Residential_school_mobility": RESIDENTIAL_SCHOOL_MOBILITY,
    "Structured_activity_involvement": STRUCTURED_ACTIVITY_INVOLVEMENT,
    "Behavioral_externalizing": sorted(set(
        IMPULSIVITY_IRRITABILITY
        + EXTERNALIZING_INITIATION_AGE
        + DELINQUENCY_VIOLENCE
    )),

    "Sleep": SLEEP_DURATION,
}

ABLATION_COMBINATIONS = {
    "Depression_plus_Bullying": [
        "Internalizing",
        "Bullying",
    ],

    "Interpersonal_context": [
        "Bullying",
        "Peer_context",
        "Family_context",
    ],

    "Psychological_plus_Interpersonal": [
        "Internalizing",
        "Bullying",
        "Peer_context",
        "Family_context",
    ],
}


DISPLAY_LABELS = {
    "Behavioral_externalizing": "Behavioral externalizing",
    "Bullying": "Bullying",
    "Community_environment": "Community environment",
    "Demographic_background": "Demographic background",
    "Depression_plus_Bullying": "Internalizing + Bullying",
    "Family_context": "Family context",
    "Internalizing": "Internalizing",
    "Interpersonal_context": "Bullying + Peer context + Family context",
    "Peer_context": "Peer context",
    "Personal_attitudes_risk_perception": "Personal attitudes / risk perception",
    "Psychological_plus_Interpersonal": "Internalizing + Bullying + Peer context + Family context",
    "Residential_school_mobility": "Residential / school mobility",
    "School_context": "School context",
    "Sleep": "Sleep",
    "Structured_activity_involvement": "Structured activity involvement",
    "Substance_use": "Substance use",
}

SELECTED_SETS = [
    "Internalizing",
    "Bullying",
    "School_context",
    "Family_context",
    "Peer_context",
    "Behavioral_externalizing",
    "Substance_use",
    "Personal_attitudes_risk_perception",
    "Community_environment",
    "Demographic_background",
    "Residential_school_mobility",
    "Structured_activity_involvement",
    "Sleep",
    "Depression_plus_Bullying",
    "Interpersonal_context",
    "Psychological_plus_Interpersonal",
]

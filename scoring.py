from config import CONSTRAINT_SCHEMA

# def compute_ci_minus_ai(ci, ai, schema):
#     if schema["type"] == "numeric":
#         return ci - ai if schema["better"] == "lower" else ai - ci

#     if schema["type"] == "ordinal":
#         return (
#             schema["levels"].index(ci)
#             - schema["levels"].index(ai)
#         )

#     if schema["type"] == "binary":
#         return int(ci) - int(ai)

#     raise ValueError("Unknown constraint type")


# def compute_score(problem, capability):
#     """
#     S = Σ w_i (c_i - a_i) + β * TRL
#     """
#     S = 0.0

#     for k, schema in CONSTRAINT_SCHEMA.items():
#         ci = problem["constraints"][k]
#         ai = capability[k]
#         wi = schema["weight"]

#         S += wi * compute_ci_minus_ai(ci, ai, schema)

#     S += problem["beta"] * capability["TRL"]
#     return S

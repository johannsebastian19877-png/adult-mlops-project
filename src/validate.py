# src/validate.py
import pandera as pa

schema = pa.DataFrameSchema({
  'age': pa.Column(int,
    checks=pa.Check.in_range(17,90)),
  'workclass': pa.Column(str,
    nullable=True),
  'education-num': pa.Column(int,
    checks=pa.Check.in_range(1,16)),
  # ... más columnas
})

# Artefacto: validation_report.json
report = schema.validate(X)

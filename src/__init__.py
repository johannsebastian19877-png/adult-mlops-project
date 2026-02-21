from ucimlrepo import fetch_ucirepo

# Obtener dataset Adult (ID=2)
adult = fetch_ucirepo(id=2)

X = adult.data.features    # DataFrame 48842 × 14
y = adult.data.targets     # DataFrame 48842 × 1

print(adult.metadata)      # Metadata del dataset
print(adult.variables)     # Info de cada variable

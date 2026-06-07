"""Configuration parameters for the forest restoration ML framework."""

RANDOM_STATE = 42

ECOLOGICAL_WEIGHTS = {
    "latitude": 0.25,
    "clima": 0.30,
    "solo": 0.30,
    "crescimento": 0.05,
    "reflorestamento": 0.05,
    "sucessao": 0.05,
}

REFOREST_CONTEXTS = [
    "matas ciliar", "encostas íngremes", "áreas degradadas", "reflorestamento misto",
    "reflorestamento encosta", "terrenos erodidos", "preservação permanente", "ecossistemas degradados",
    "cursos d´água", "revegetação", "solos compactos", "recuperação de solo", "margens de rio",
    "ciliar com inundação", "mata ciliar sem inundações", "mata ciliar com ou sem inundações",
    "solos de baixa fertilidade", "terrenos depauperados", "áreas com solo permanentemente encharcado",
    "recuperação de solos pouco férteis", "estabilização de dunas", "terrenos queimados",
    "áreas erodidas", "áreas mineração", "solos erodidos", "drenagem lenta",
    "margens desmatadas", "margem de represa com piscicultura",
]

SOIL_TERMS = [
    "arenoso", "argiloso", "humoso", "calcario", "areno-argilosa",
    "franco-argilosa", "franco-arenosa", "seco", "umido",
    "baixa", "media", "alta", "fertil", "depauperado",
]

ML_TEXT_COLUMNS = [
    "dispersao", "densidade", "solo", "caracteristicas_silviculturais",
    "crescimento_producao", "contextos_reflorestamento", "medicinal", "oleo",
    "resina", "tipos_climaticos",
]

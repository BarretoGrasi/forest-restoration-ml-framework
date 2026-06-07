import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------
# CONFIGURAÇÕES E PESOS MODULARIZADOS
# -------------------------
CSV_PATH = "/content/dataset_floresta2.csv"
RANDOM_STATE = 42

# POLÍTICA F: Foco Edáfico-Climático (85%) + Crescimento (10%) + Reflorestamento (5%)
PESOS_RANQUEAMENTO_ECOLOGICO = {
    'latitude': 0.25,
    'clima': 0.30,
    'solo': 0.30,
    'crescimento': 0.05,
    'reflorestamento': 0.05,
    'sucessao': 0.05
}

# -------------------------
# LISTA DE CONTEXTOS DE REFLORESTAMENTO
# -------------------------
REFOREST_CONTEXTS = [
    "matas ciliar", "encostas íngremes", "áreas degradadas", "reflorestamento misto",
    "reflorestamento encosta", "terrenos erodidos", "preservação permanente", "ecossistemas degradados",
    "cursos d´água", "revegetação", "solos compactos","recuperação de solo", "margens de rio",
    "ciliar com inundação", "mata ciliar sem inundações", "mata ciliar com ou sem inundações",
    "solos de baixa fertilidade", "terrenos depauperados", "áreas com solo permanentemente encharcado",
    "recuperação de solos pouco férteis", "estabilização de dunas",
    "terrenos queimados", "áreas erodidas", "áreas mineração",
    "solos erodidos", "drenagem lenta", "margens desmatadas", "margem de represa com piscicultura"
]

# -------------------------
# FUNÇÕES AUXILIARES
# -------------------------
def limpar_texto(texto):
    if pd.isna(texto): return ''
    texto = str(texto).lower()
    texto = re.sub(r'[^a-zà-ú0-9\s,;()-]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def converter_latitude(texto):
    if pd.isna(texto) or texto == '': return np.nan
    try:
        return float(re.findall(r'-?\d+\.?\d*', str(texto).lower().replace(',', '.').strip())[0])
    except:
        return np.nan

def clima_por_latitude(lat):
    if pd.isna(lat): return 'Outro'
    if -10 <= lat <= 10: return 'Af'
    elif -30 <= lat <= -19: return 'Cwa'
    elif -27 <= lat <= -20: return 'Cfa'
    elif -35 <= lat <= -25: return 'Cfb'
    return 'Outro'

def extrair_codigos_koeppen(texto):
    if pd.isna(texto) or texto == '': return []
    codigos = re.findall(r'[A-Z][a-z]{1,2}', str(texto), re.IGNORECASE)
    return [c.upper() for c in codigos]

def categorizar_densidade(dens_texto):
    if pd.isna(dens_texto) or dens_texto == '': return ''
    try:
        dens_num = float(dens_texto)
        if dens_num > 0.70: return 'alta'
        elif dens_num >= 0.50: return 'media'
        else: return 'baixa'
    except:
        return ''

# FUNÇÃO CRÍTICA AJUSTADA (V9) - SÓ CLASSIFICA COMO PIONEIRA SE A PALAVRA ESTIVER LÁ.
def resumir_sucessional(texto):
    texto_limpo = str(texto).lower()
    grupos = set()

    # 1. CORREÇÃO: Busca por Pioneira EXATA
    if 'pioneira' in texto_limpo or 'pion.' in texto_limpo:
        grupos.add('Pioneira')

    # 2. Busca por Secundária Inicial
    if 'secundaria inicial' in texto_limpo or 'secundaria in.' in texto_limpo or 'secundaria, inicial' in texto_limpo:
        grupos.add('Secundária Inicial')

    # 3. Busca por Secundária Tardia
    if 'secundaria tardia' in texto_limpo or 'secundaria final' in texto_limpo or 'tardia' in texto_limpo:
        grupos.add('Secundária Tardia')

    # 4. Busca por Clímax
    if 'climax' in texto_limpo or 'clímace' in texto_limpo or 'clímax' in texto_limpo or 'clã­max' in texto_limpo:
        grupos.add('Clímax')

    # Heurística para o termo 'inicial' solto: Assumimos Secundária Inicial se for a única menção.
    if 'inicial' in texto_limpo and not any(g in grupos for g in ['Pioneira', 'Secundária Inicial', 'Secundária Tardia', 'Clímax']):
        grupos.add('Secundária Inicial')

    grupos_list = list(grupos)

    ordem = {'Pioneira': 1, 'Secundária Inicial': 2, 'Secundária Tardia': 3, 'Clímax': 4}
    grupos_list.sort(key=lambda x: ordem.get(x, 5))

    return ', '.join(grupos_list) if grupos_list else 'Indefinido'

def extrair_contextos_reflorestamento(texto):
    texto = limpar_texto(texto)
    encontrados = [ctx for ctx in REFOREST_CONTEXTS if ctx in texto]
    return ', '.join(encontrados) if encontrados else 'indefinido'

def limpar_texto_exibicao(texto):
    if pd.isna(texto): return 'N/A'
    texto = str(texto)
    try:
        texto = texto.encode('latin1').decode('utf-8', 'ignore')
    except:
        pass
    texto = re.sub(r'[\r\n\t]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    if len(texto) > 100:
        texto = texto[:97] + '...'
    return texto

def extrair_densidade_populacional(texto):
    if pd.isna(texto) or texto == '': return 0.0
    texto = str(texto).lower().replace('por hectare', 'ha').replace('indivíduos', 'ind')
    matches = re.findall(r'(\d+)\s+(?:arvore|arvores|ind|ha|individuos)', texto)
    if matches:
        valores = [int(m) for m in matches if m.isdigit()]
        if valores:
            return np.mean(valores)
    return 0.0

# FUNÇÃO DE PONTUAÇÃO ECOLÓGICA (Mantida da V8)
def pontuacao_ecologica(row, perfil_processado, pesos):

    PESO_LATITUDE = pesos['latitude']
    PESO_CLIMA = pesos['clima']
    PESO_SOLO = pesos['solo']
    PESO_SUCESSAO = pesos['sucessao']
    PESO_REFLORESTAMENTO = pesos['reflorestamento']
    PESO_CRESCIMENTO = pesos.get('crescimento', 0.0)

    # 1. Latitude
    lat_especie = row['latitude_num']
    lat_perfil = perfil_processado['latitude']
    compat_lat = 1 - (abs(lat_especie - lat_perfil) / 40) if not pd.isna(lat_especie) else 0.5
    compat_lat = max(0, min(1, compat_lat))

    # 2. Clima
    clima_especie_codigos = extrair_codigos_koeppen(row['tipos_climaticos'])
    clima_perfil_codigo = perfil_processado['tipos_climaticos'].upper()

    if clima_perfil_codigo in clima_especie_codigos: compat_clima = 1.0
    elif not clima_especie_codigos or clima_perfil_codigo == 'OUTRO': compat_clima = 0.5
    else: compat_clima = 0.2

    # 3. Solo
    TERMOS_SOLO = ["arenoso", "argiloso", "humoso", "calcario", "areno-argilosa",
                    "franco-argilosa", "franco-arenosa", "seco", "umido",
                    "baixa", "media", "alta", "fertil", "depauperado"]
    solo_perfil_limpo = perfil_processado['solo']
    solo_especie_limpo = row['solo']
    compat_solo = 0.5
    termos_perfil_presentes = [t for t in TERMOS_SOLO if t in solo_perfil_limpo]
    if termos_perfil_presentes:
        compat_solo = 1.0 if any(t in solo_especie_limpo for t in termos_perfil_presentes) else 0.7

    # 4. Sucessão
    grupo_raw = row['grupo_sucessional_resumido'].lower()
    compat_sucessao = 1.0 if any(g in grupo_raw for g in ['pioneira','secundaria inicial','secundaria tardia','climax']) else 0.6

    # 5. Reflorestamento
    contextos_especie = row['contextos_reflorestamento']
    contextos_perfil = perfil_processado['contextos_reflorestamento']
    compat_reflorestamento = 0.6
    if contextos_perfil != 'indefinido':
        contextos_perfil_list = [c.strip() for c in contextos_perfil.split(', ') if c.strip()]
        if any(ctx in contextos_especie for ctx in contextos_perfil_list):
             compat_reflorestamento = 1.0

    # 6. Crescimento
    cresc_perfil = perfil_processado.get('crescimento_producao', '').lower()
    cresc_especie = row['crescimento_producao'].lower()
    compat_crescimento = 0.5

    if 'rapido' in cresc_perfil:
        if 'rapido' in cresc_especie: compat_crescimento = 1.0
        elif 'moderado' in cresc_especie: compat_crescimento = 0.7
        else: compat_crescimento = 0.3
    elif 'lento' in cresc_perfil:
        if 'lento' in cresc_especie: compat_crescimento = 1.0
        elif 'moderado' in cresc_especie: compat_crescimento = 0.7
        else: compat_crescimento = 0.3
    else:
        if 'moderado' in cresc_especie: compat_crescimento = 1.0
        elif 'rapido' in cresc_especie or 'lento' in cresc_especie: compat_crescimento = 0.7


    # CÁLCULO FINAL
    pont_final = (
        PESO_LATITUDE * compat_lat +
        PESO_CLIMA * compat_clima +
        PESO_SOLO * compat_solo +
        PESO_SUCESSAO * compat_sucessao +
        PESO_REFLORESTAMENTO * compat_reflorestamento +
        PESO_CRESCIMENTO * compat_crescimento
    )

    return pont_final, {
        'latitude': compat_lat, 'clima': compat_clima, 'solo': compat_solo,
        'sucessao': compat_sucessao,
        'reflorestamento': compat_reflorestamento,
        'crescimento': compat_crescimento
    }

# -------------------------
# CARREGAR E PRÉ-PROCESSAR DADOS
# -------------------------
try:
    df_raw = pd.read_csv(CSV_PATH, sep=';', encoding='latin1', dtype=str)
except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em {CSV_PATH}. Verifique o caminho.")
    exit()

df_raw.columns = [c.strip() for c in df_raw.columns]

# Mapeamento e limpeza de colunas
df = pd.DataFrame()
df['nome'] = df_raw['nome']
df['nome_cientifico'] = df_raw.get('nome_cientifico', df_raw['nome'])
df['latitude'] = df_raw['latitude']
df['grupo_sucessional'] = df_raw['grupo_sucessional']
df['densidade'] = df_raw['densidade']
df['tipos_climaticos'] = df_raw.get('tipos_climaticos (koeppen)', df_raw.get('tipos_climaticos', pd.NA))
df['solo'] = df_raw['solo']
df['reflorestamento_original'] = df_raw.get('reflorestamento_recuperacao ambiental:', df_raw.get('reflorestamento_recuperacao_ambiental', pd.NA))
df['dispersao'] = df_raw.get('dispersão de frutos e sementes', df_raw.get('dispersao', pd.NA))
df['caracteristicas_silviculturais'] = df_raw.get('características silviculturais', df_raw.get('caracteristicas_silviculturais', pd.NA))
# Correção do nome da coluna para mapeamento
df['crescimento_producao'] = df_raw.get('crescimento_produção', df_raw.get('crescimento', pd.NA))
df['medicinal'] = df_raw.get('medicinal', pd.NA)
df['oleo'] = df_raw.get('oleo', pd.NA)
df['resina'] = df_raw.get('resina', pd.NA)

# -> FILTRAGEM DE LINHAS COM NOMES FALTANDO
df = df[df['nome'].notna() & (df['nome'] != '')]
df['nome_cientifico'] = df['nome_cientifico'].replace(r'^\s*$', np.nan, regex=True)
df = df[df['nome_cientifico'].notna()]
df = df.reset_index(drop=True)

# Aplicação da limpeza de texto nas colunas (Corrigido para evitar AttributeError)
text_cols = ['dispersao','densidade','solo','caracteristicas_silviculturais',
             'crescimento_producao','reflorestamento_original','medicinal','oleo','resina']
for c in text_cols:
    df[c] = df[c].apply(limpar_texto)

df['latitude_num'] = df['latitude'].apply(converter_latitude)
df['tipos_climaticos'] = df.apply(
    lambda r: clima_por_latitude(r['latitude_num'])
    if pd.isna(r['tipos_climaticos']) or r['tipos_climaticos']==''
    else r['tipos_climaticos'], axis=1)

df['densidade'] = df['densidade'].apply(categorizar_densidade)
# AQUI USA A FUNÇÃO CORRIGIDA
df['grupo_sucessional_resumido'] = df['grupo_sucessional'].apply(resumir_sucessional)
df['contextos_reflorestamento'] = df['reflorestamento_original'].apply(extrair_contextos_reflorestamento)
df['densidade_populacional'] = df['reflorestamento_original'].apply(extrair_densidade_populacional)

# -------------------------
# TREINAMENTO ML (TARGET ECOLÓGICO REAL)
# -------------------------
perfil_base_raw = {
    'latitude': -34.0, 'solo': 'franco-argilosa, fertil, umido', 'caracteristicas_silviculturais': 'sombreamento',
    'crescimento_producao': 'moderado', 'densidade': '0.15', 'reflorestamento': 'matas ciliar, areas degradadas',
    'medicinal': 'sim', 'oleo': 'sim', 'resina': 'sim', 'madeira_serrada': 'sim', 'energia':'sim','constituintes_quimicos':'sim',
    'substancias_tanantes':'sim','celulose_papel':'sim','goma':'sim','forrageiro':'sim','cortica':'sim','corante':'sim','':'sim',
    'alimentacao_humana':'sim','fibras_mucilagens':'sim','perfume':'sim','sabao ':'sim','paina':'sim','vime':'sim',
    'sapopinas':'sim','alimentacao_animal':'sim','inseticida':'sim','alimentacao_animal':'sim','cera':'sim','cosmetico':'sim','hormonio':'sim',
    'proteinas':'sim','fibras':'sim','cumarina':'sim','condimento':'sim','latex':'sim','cordoaria':'sim','biodisel':'sim',' aplicacoes_industriais':'sim',
    'artesanais':'sim','apicola':'sim','paisagistico':'sim'}

PESOS_ML_TREINAMENTO = PESOS_RANQUEAMENTO_ECOLOGICO.copy()

perfil_base_limpo = {k: v for k,v in perfil_base_raw.items()}
perfil_base_limpo['solo'] = limpar_texto(perfil_base_limpo['solo'])
perfil_base_limpo['densidade'] = categorizar_densidade(str(perfil_base_limpo['densidade']))
perfil_base_limpo['tipos_climaticos'] = clima_por_latitude(perfil_base_raw['latitude'])
perfil_base_limpo['contextos_reflorestamento'] = extrair_contextos_reflorestamento(perfil_base_limpo['reflorestamento'])
perfil_base_limpo['densidade_populacional'] = extrair_densidade_populacional(perfil_base_limpo['reflorestamento'])
perfil_base_limpo['crescimento_producao'] = limpar_texto(perfil_base_limpo['crescimento_producao'])

df['pontuacao_target'] = df.apply(lambda r: pontuacao_ecologica(r, perfil_base_limpo, PESOS_ML_TREINAMENTO)[0], axis=1)
y = df['pontuacao_target'].values

ml_cols_train = ['dispersao','densidade','solo','caracteristicas_silviculturais','crescimento_producao',
                 'contextos_reflorestamento','medicinal','oleo','resina','tipos_climaticos']
df['features_texto'] = df[ml_cols_train].astype(str).agg(' '.join, axis=1)

tfidf = TfidfVectorizer(max_features=800, ngram_range=(1,2))
X_text = tfidf.fit_transform(df['features_texto']).toarray()
X_num = df[['densidade_populacional']].values
X = np.hstack((X_text, X_num))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
modelo = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE)
modelo.fit(X_train, y_train)

# -------------------------
# PERFIL DO LOCAL (ENTRADA DO USUÁRIO)
# -------------------------
'''perfil_usuario_raw = {
    'latitude': -24.25, 'solo': 'secos pedregosos', 'caracteristicas_silviculturais': 'mata ciliares',
    'crescimento_producao': 'moderado',
    'densidade': '0.1', 'reflorestamento': 'mata ciliar',
    'medicinal': 'sim', 'oleo': 'sim', 'resina': 'sim'
}

perfil_usuario_limpo = {k: v for k,v in perfil_usuario_raw.items()}
perfil_usuario_limpo['solo'] = limpar_texto(perfil_usuario_limpo['solo'])
perfil_usuario_limpo['densidade'] = categorizar_densidade(str(perfil_usuario_limpo['densidade']))
perfil_usuario_limpo['tipos_climaticos'] = clima_por_latitude(perfil_usuario_raw['latitude'])
perfil_usuario_limpo['contextos_reflorestamento'] = extrair_contextos_reflorestamento(perfil_usuario_limpo['reflorestamento'])
perfil_usuario_limpo['densidade_populacional'] = extrair_densidade_populacional(perfil_usuario_raw['reflorestamento'])
perfil_usuario_limpo['crescimento_producao'] = limpar_texto(perfil_usuario_limpo['crescimento_producao'])

perfil_texto_usuario = ' '.join([str(perfil_usuario_limpo.get(c,'')) for c in ml_cols_train])
perfil_tfidf_usuario = tfidf.transform([perfil_texto_usuario]).toarray()
perfil_num_usuario = np.array([[perfil_usuario_limpo['densidade_populacional']]])
perfil_features_usuario = np.hstack((perfil_tfidf_usuario, perfil_num_usuario))'''

perfil_usuario_raw = {
    # 1. Dados Ecológicos/Silviculturais (Entrada do Usuário)
    'latitude': -34.25,
    'solo': 'umidos pedregosos',
    'caracteristicas_silviculturais': 'sombreamento',
    'crescimento_producao': 'lento',
    'densidade': '0.15',
    'reflorestamento': 'áreas degradadas',

    # 2. Usos Desejados (Entrada do Usuário no Exemplo)
    'medicinal': 'sim',
    'oleo': 'sim',
    'resina': 'sim',
    'substancias_tanantes':'sim',

    # 3. Outros Usos (Padrão: N/A - Neutro para manter a estrutura ML)
    'madeira_serrada': 'N/A', 'energia':'N/A','constituintes_quimicos':'N/A',
    'celulose_papel':'N/A','goma':'N/A','forrageiro':'N/A',
    'cortica':'N/A','corante':'N/A','alimentacao_humana':'N/A',
    'fibras_mucilagens':'N/A','perfume':'N/A','sabao':'N/A','paina':'N/A',
    'vime':'N/A','sapopinas':'N/A','alimentacao_animal':'N/A','inseticida':'N/A',
    'cera':'N/A','cosmetico':'N/A','hormonio':'N/A','proteinas':'N/A','fibras':'N/A',
    'cumarina':'N/A','condimento':'N/A','latex':'N/A','cordoaria':'N/A','biodisel':'N/A',
    'aplicacoes_industriais':'N/A','artesanais':'N/A','apicola':'N/A','paisagistico':'N/A'
}


perfil_usuario_limpo = {k: v for k,v in perfil_usuario_raw.items()}
perfil_usuario_limpo['solo'] = limpar_texto(perfil_usuario_limpo['solo'])
perfil_usuario_limpo['densidade'] = categorizar_densidade(str(perfil_usuario_limpo['densidade']))
perfil_usuario_limpo['tipos_climaticos'] = clima_por_latitude(perfil_usuario_raw['latitude'])
perfil_usuario_limpo['contextos_reflorestamento'] = extrair_contextos_reflorestamento(perfil_usuario_limpo['reflorestamento'])
perfil_usuario_limpo['densidade_populacional'] = extrair_densidade_populacional(perfil_usuario_raw['reflorestamento'])
perfil_usuario_limpo['crescimento_producao'] = limpar_texto(perfil_usuario_limpo['crescimento_producao'])

perfil_texto_usuario = ' '.join([str(perfil_usuario_limpo.get(c,'')) for c in ml_cols_train])
perfil_tfidf_usuario = tfidf.transform([perfil_texto_usuario]).toarray()
perfil_num_usuario = np.array([[perfil_usuario_limpo['densidade_populacional']]])
perfil_features_usuario = np.hstack((perfil_tfidf_usuario, perfil_num_usuario))


# -------------------------
# GERAÇÃO DA PONTUAÇÃO HÍBRIDA
# -------------------------
y_pred_test = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

X_all_text = tfidf.transform(df['features_texto']).toarray()
X_all_num = df[['densidade_populacional']].values
X_all = np.hstack((X_all_text, X_all_num))
df['pontuacao_ml_prevista'] = modelo.predict(X_all)

# Ranqueamento Ecológico: Usando o perfil do usuário e os pesos da POLÍTICA F
df['pontuacao_ecologica_final'] = df.apply(lambda r: pontuacao_ecologica(r, perfil_usuario_limpo, PESOS_RANQUEAMENTO_ECOLOGICO)[0], axis=1)

PESO_ECOLOGICO = 0.5
PESO_ML = 0.5
df['pontuacao_hibrida'] = (df['pontuacao_ecologica_final'] * PESO_ECOLOGICO) + (df['pontuacao_ml_prevista'] * PESO_ML)

compatibilidade_prevista = modelo.predict(perfil_features_usuario)[0]

# -------------------------
# RESULTADOS FINAIS E EXIBIÇÃO
# -------------------------
top_especies = df.sort_values('pontuacao_hibrida', ascending=False).head(10).copy()
melhor = df.loc[df['pontuacao_hibrida'].idxmax()]
detalhes = pontuacao_ecologica(melhor, perfil_usuario_limpo, PESOS_RANQUEAMENTO_ECOLOGICO)[1]

# Formatação de Saída
top_especies['Uso Primário Registrado'] = top_especies['reflorestamento_original'].apply(limpar_texto_exibicao)
top_especies['Grupo Sucessional'] = top_especies['grupo_sucessional_resumido']

pd.set_option('display.max_colwidth', 60)

print("="*110)
print("RESULTADOS DA COMPATIBILIDADE DE ESPÉCIES FLORESTAIS (POLÍTICA F: Cresc. e Refl. Ativos - V9)")
print("="*110)
print(f"Latitude do perfil: {perfil_usuario_raw['latitude']} | Clima: {perfil_usuario_limpo['tipos_climaticos']}")
print(f"Solo do perfil: {perfil_usuario_raw['solo'].title()} | Crescimento Prioritário: {perfil_usuario_raw['crescimento_producao'].title()}")
print(f"Foco(s) de Reflorestamento: {perfil_usuario_limpo['contextos_reflorestamento'].title()}")
print("--------------------------------------------------------------------------------------------------")
print(f"PESOS ATIVOS: Clima={PESOS_RANQUEAMENTO_ECOLOGICO['clima']:.2f}, Solo={PESOS_RANQUEAMENTO_ECOLOGICO['solo']:.2f}, Crescimento={PESOS_RANQUEAMENTO_ECOLOGICO['crescimento']:.2f}, Reflorestamento={PESOS_RANQUEAMENTO_ECOLOGICO['reflorestamento']:.2f}")
print(f"Desempenho do Modelo ML (R2/RMSE): R²={r2:.3f}, RMSE={rmse:.3f} (Target Ecológico)")
print(f"Compatibilidade Prevista pelo Modelo ML (Input do Usuário): {compatibilidade_prevista:.3f}")
print("--------------------------------------------------------------------------------------------------")
print("Pontuação Ecológica Detalhada (Melhor Espécie):")
for fator, valor in detalhes.items():
    peso = PESOS_RANQUEAMENTO_ECOLOGICO.get(fator, 0.00)
    peso_info = f"(Peso {peso:.2f})"
    print(f"  • {fator.capitalize():<15}: {valor:.3f} {peso_info}")
print(f"  → Pontuação Híbrida Final: {melhor['pontuacao_hibrida']:.3f}")
print("--------------------------------------------------------------------------------------------------")
print("Top 10 Espécies Recomendadas (Ordenadas por Pontuação Híbrida):")

top_display = top_especies[['nome', 'nome_cientifico', 'Grupo Sucessional', 'pontuacao_hibrida', 'Uso Primário Registrado']]
top_display.columns = ['Espécie', 'Nome Científico', 'Grupo Sucessional', 'Pontuação Híbrida', 'Uso Primário Registrado']

print(top_display.to_string(index=False))
print("="*110)
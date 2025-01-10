pkg load statistics;

% ========================= Carregar Dados =========================
disp("Carregando os dados...");
% Alterar para o local onde estão salvos os dados
disp("Carregando os dados_base...");
dados_base = dlmread('C:/Users/ferna/Desktop/Inteligência Artificial 1/IA-trabalho/Dados_Base.csv', ',');
disp("Carregando as classes_base...");
classes_base = textread('C:/Users/ferna/Desktop/Inteligência Artificial 1/IA-trabalho/Classes_Base.csv', '%s', 'delimiter', ',');
disp("Carregando os dados_novos...");
dados_novos = dlmread('C:/Users/ferna/Desktop/Inteligência Artificial 1/IA-trabalho/Dados_Novos.csv', ',');
disp("Concluído!");

% ========================= Representação Gráfica Dados-Base =========================
disp("Plotando os dados base...");
figure;
scatter(dados_base(:, 1), dados_base(:, 2), 50, 'filled');
title("Representação Gráfica dos Dados Base");
xlabel("Feature 1");
ylabel("Feature 2");
grid on;


% ========================= Normalização =========================
disp("Normalizando os dados_base...");
dados_base_norm = zscore(dados_base);
disp("Concluído!");

disp("Normalizando os dados_novos...");
dados_novos_norm = zscore(dados_novos);
disp("Concluído!");


% ========================= Transformação das Classes para Formato Binário =========================
[classes_unicas, ~, classes_numericas] = unique(classes_base);
num_classes = length(classes_unicas);
num_amostras = length(classes_base);
classes_binarias = eye(num_classes)(classes_numericas, :);

% ========================= Oversampling =========================
disp("Oversampling Iniciado..");
contagem_classes = hist(classes_numericas, num_classes);
min_class_count = min(contagem_classes);
min_classes_idx = find(contagem_classes == min_class_count);
dados_base_balanciado = dados_base_norm;
classes_base_balanciado = classes_numericas;

for i = 1:length(min_classes_idx)
    min_class_idx = min_classes_idx(i);

    amostras_minoritaria = dados_base_norm(classes_numericas == min_class_idx, :);

    num_amostras_faltando = max(contagem_classes) - min_class_count;

    num_amostras_faltando = min(num_amostras_faltando, size(amostras_minoritaria, 1));

    amostras_duplicadas = amostras_minoritaria(randperm(size(amostras_minoritaria, 1), num_amostras_faltando), :);

    dados_base_balanciado = [dados_base_balanciado; amostras_duplicadas];
    classes_base_balanciado = [classes_base_balanciado; repmat(min_class_idx, num_amostras_faltando, 1)];
end

disp("Oversampling concluído para todas as classes minoritárias!");


% ========================= PCA - Dados-Base =========================
disp("Realizando PCA...");
% Centralizar os dados
dados_base_centralizados = dados_base_norm - mean(dados_base_norm);
% Calcular a matriz de covariância
covariancia = cov(dados_base_centralizados);
% Calcular os autovalores e autovetores
[autovetores, autovalores] = eig(covariancia);
% Ordenar os autovalores e selecionar as componentes principais
[autovalores, indice] = sort(diag(autovalores), 'descend');
autovetores = autovetores(:, indice);
pca = 2000;  % número de componentes principais desejadas
dados_reduzidos = dados_base_centralizados * autovetores(:, 1:pca);
disp("PCA concluído!");


% ========================= Visualizar os dados projetados =========================
figure;
scatter(dados_reduzidos(:, 1), dados_reduzidos(:, 2), 10, classes_numericas, 'filled');
title('Projeção dos Dados com PCA');
xlabel('Primeira Componente Principal');
ylabel('Segunda Componente Principal');


% ========================= Validação Cruzada k-Fold =========================
k = 3;  % Número de folds
disp(['Executando validação cruzada com ', num2str(k), ' folds...']);

% Gerar índices manualmente
indices = repmat(1:k, 1, ceil(num_amostras / k));
indices = indices(1:num_amostras);
indices = indices(randperm(num_amostras));  % Embaralhar os índices

f1_scores_fold = zeros(1, k);

for fold = 1:k
    disp(['Fold ', num2str(fold), ' de ', num2str(k)]);

    % Divisão dos dados em treinamento e teste
    teste_idx = (indices == fold);
    treino_idx = ~teste_idx;

    dados_treinamento = dados_base_balanciado(treino_idx, :);
    classes_treinamento = classes_base_balanciado(treino_idx);

    dados_teste = dados_base_balanciado(teste_idx, :);
    classes_teste = classes_base_balanciado(teste_idx);

    % Inicializar rede neural (MLP)
    num_neuronios_ocultos = 10;
    num_epocas = 1000;
    taxa_aprendizado = 0.01;

    tamanho_entrada = size(dados_treinamento, 2);
    tamanho_saida = num_classes;
    pesos_entrada_oculta = rand(tamanho_entrada, num_neuronios_ocultos) * 0.01;
    bias_oculta = zeros(1, num_neuronios_ocultos);
    pesos_oculta_saida = rand(num_neuronios_ocultos, tamanho_saida) * 0.01;
    bias_saida = zeros(1, tamanho_saida);

    sigmoid = @(x) 1 ./ (1 + exp(-x));
    sigmoid_derivada = @(x) sigmoid(x) .* (1 - sigmoid(x));

    % Treinamento
    for epoca = 1:num_epocas
        camada_oculta = sigmoid(dados_treinamento * pesos_entrada_oculta + bias_oculta);
        camada_saida = sigmoid(camada_oculta * pesos_oculta_saida + bias_saida);

        erro = camada_saida - eye(num_classes)(classes_treinamento, :);

        delta_saida = erro .* sigmoid_derivada(camada_saida);
        delta_oculta = (delta_saida * pesos_oculta_saida') .* sigmoid_derivada(camada_oculta);

        pesos_oculta_saida -= taxa_aprendizado * (camada_oculta' * delta_saida);
        bias_saida -= taxa_aprendizado * sum(delta_saida, 1);

        pesos_entrada_oculta -= taxa_aprendizado * (dados_treinamento' * delta_oculta);
        bias_oculta -= taxa_aprendizado * sum(delta_oculta, 1);
    end

    % Validação
    camada_oculta_teste = sigmoid(dados_teste * pesos_entrada_oculta + bias_oculta);
    camada_saida_teste = sigmoid(camada_oculta_teste * pesos_oculta_saida + bias_saida);

    [~, classe_prevista] = max(camada_saida_teste, [], 2);

    % Cálculo do F1-Score
    f1_scores = zeros(1, num_classes);
    for i = 1:num_classes
        classes_binarias_reais = (classes_teste == i);
        classes_binarias_previstas = (classe_prevista == i);

        precisao = sum(classes_binarias_reais & classes_binarias_previstas) / (sum(classes_binarias_previstas) + eps);
        recall = sum(classes_binarias_reais & classes_binarias_previstas) / (sum(classes_binarias_reais) + eps);

        f1_scores(i) = 2 * (precisao * recall) / (precisao + recall + eps);
    end

    f1_scores_fold(fold) = mean(f1_scores);

end

% Resultado final
f1_score_medio = mean(f1_scores_fold);
disp(['F1-Score médio em validação cruzada: ', num2str(f1_score_medio)]);


% ========================= Classificação dos Dados Novos =========================
disp("Classificando os dados novos...");

% Forward pass para os dados novos
camada_oculta_novos = sigmoid(dados_novos_norm * pesos_entrada_oculta + bias_oculta);
camada_saida_novos = sigmoid(camada_oculta_novos * pesos_oculta_saida + bias_saida);

[~, classes_previstas_novos] = max(camada_saida_novos, [], 2);

disp("Classificação concluída!");
disp("Classes previstas para os dados novos:");
disp(classes_previstas_novos);



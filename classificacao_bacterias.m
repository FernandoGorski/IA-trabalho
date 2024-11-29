pkg load statistics;

% ========================= Carregar Dados =========================
disp("Carregando os dados...");
% Alterar para o local onde estão salvos os dados
% dados_base = dlmread('/IA-trabalho/Dados_Base.csv', ',');
% classes_base = textread('/IA-trabalho/Classes_Base.csv', '%s', 'delimiter', ',');
% dados_novos = dlmread('/IA-trabalho/Dados_Novos.csv', ',');
disp("Concluído!");

% ========================= Normalização =========================
% Normalizar os dados da base e os dados novos
disp("Normalizando os dados_base...");
dados_base_norm = zscore(dados_base);
disp("Concluído!");

disp("Normalizando os dados_novos...");
dados_novos_norm = zscore(dados_novos);
disp("Concluído!");

% ========================= PCA =========================
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
k = 10;  % número de componentes principais desejadas
dados_reduzidos = dados_base_centralizados * autovetores(:, 1:k);
disp("PCA concluído!");

% Visualizar os dados projetados em 2D (se k >= 2)
figure;
scatter(dados_reduzidos(:, 1), dados_reduzidos(:, 2), 10, classes_numericas, 'filled');
title('Projeção dos Dados com PCA');
xlabel('Primeira Componente Principal');
ylabel('Segunda Componente Principal');

% ========================= Transformação das Classes para Formato Binário =========================
[classes_unicas, ~, classes_numericas] = unique(classes_base);
num_classes = length(classes_unicas);  % Número de classes
num_amostras = length(classes_base);
classes_binarias = eye(num_classes)(classes_numericas, :);

% ========================= Divisão dos Dados em Treinamento e Teste =========================
disp("Dividindo os dados em treinamento e teste...");
n_amostras = size(dados_reduzidos, 1);  % Total de amostras
num_treinamento = round(0.8 * n_amostras);  % 80% para treinamento

% Embaralhar os índices das amostras
indices = randperm(n_amostras);

% Selecionar índices para treinamento e teste
indice_treinamento = indices(1:num_treinamento);
indice_teste = indices(num_treinamento+1:end);

% Dividir os dados
dados_treinamento = dados_reduzidos(indice_treinamento, :);
classes_treinamento = classes_numericas(indice_treinamento);
dados_teste = dados_reduzidos(indice_teste, :);
classes_teste = classes_base(indice_teste);

disp("Divisão concluída!");

% ========================= Criar a Rede Neural (MLP) =========================
disp("Criando a rede neural...");
% Definir parâmetros da rede neural
num_neuronios_ocultos = 10;  % Número de neurônios na camada oculta
num_epocas = 1000;  % Número de épocas de treinamento
taxa_aprendizado = 0.01;  % Taxa de aprendizado

% Inicializar pesos e bias
tamanho_entrada = size(dados_treinamento, 2);  % Número de características
tamanho_saida = num_classes;  % Número de classes
pesos_entrada_oculta = rand(tamanho_entrada, num_neuronios_ocultos) * 0.01;
bias_oculta = zeros(1, num_neuronios_ocultos);
pesos_oculta_saida = rand(num_neuronios_ocultos, tamanho_saida) * 0.01;
bias_saida = zeros(1, tamanho_saida);

% Função de ativação sigmoid
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoid_derivada = @(x) sigmoid(x) .* (1 - sigmoid(x));

% ========================= Treinamento da Rede =========================
for epoca = 1:num_epocas
    % Forward pass
    camada_oculta = sigmoid(dados_treinamento * pesos_entrada_oculta + bias_oculta);
    camada_saida = sigmoid(camada_oculta * pesos_oculta_saida + bias_saida);

    % Cálculo do erro
    erro = camada_saida - eye(num_classes)(classes_treinamento, :);

    % Backpropagation
    delta_saida = erro .* sigmoid_derivada(camada_saida);
    delta_oculta = (delta_saida * pesos_oculta_saida') .* sigmoid_derivada(camada_oculta);

    % Atualização dos pesos e bias
    pesos_oculta_saida -= taxa_aprendizado * (camada_oculta' * delta_saida);
    bias_saida -= taxa_aprendizado * sum(delta_saida, 1);

    pesos_entrada_oculta -= taxa_aprendizado * (dados_treinamento' * delta_oculta);
    bias_oculta -= taxa_aprendizado * sum(delta_oculta, 1);

    % Exibir erro médio por época
    if mod(epoca, 100) == 0
        erro_medio = mean(abs(erro(:)));
        disp(['Época ', num2str(epoca), ': Erro médio = ', num2str(erro_medio)]);
    end
end

% ========================= Validação nos Dados de Teste =========================
disp("Validando nos dados de teste...");
camada_oculta_teste = sigmoid(dados_teste * pesos_entrada_oculta + bias_oculta);
camada_saida_teste = sigmoid(camada_oculta_teste * pesos_oculta_saida + bias_saida);

% Converter predições em classes
[~, classe_prevista] = max(camada_saida_teste, [], 2);
[~, ~, classes_teste] = unique(classes_teste);

% ========================= Classificação dos Novos Dados =========================
disp("Classificando os novos dados...");
camada_oculta_novos = sigmoid(dados_novos_norm * pesos_entrada_oculta + bias_oculta);
camada_saida_novos = sigmoid(camada_oculta_novos * pesos_oculta_saida + bias_saida);

% Converter as previsões das classes em formato numérico
[~, classes_previstas_novos] = max(camada_saida_novos, [], 2);

% ========================= Salvar os Resultados em Arquivo CSV =========================
disp("Salvando os resultados em arquivo CSV...");
resultado = [dados_novos, classes_previstas_novos];  % Adicionar as classes previstas aos dados originais
csvwrite('resultado_classificacao.csv', resultado);  % Salvar em arquivo CSV
disp("Arquivo salvo com sucesso!");


% ========================= Métricas de Desempenho =========================
% Função para calcular a precisão
function precisao = calcular_precisao(classes_reais, classes_previstas)
  tp = sum(classes_reais == 1 & classes_previstas == 1);  % Verdadeiros positivos
  fp = sum(classes_reais == 0 & classes_previstas == 1);  % Falsos positivos
  precisao = tp / (tp + fp);  % Fórmula da precisão
end

% Função para calcular o recall
function recall = calcular_recall(classes_reais, classes_previstas)
  tp = sum(classes_reais == 1 & classes_previstas == 1);  % Verdadeiros positivos
  fn = sum(classes_reais == 1 & classes_previstas == 0);  % Falsos negativos
  recall = tp / (tp + fn);  % Fórmula do recall
end

% Calcular F1-Score
f1_scores = zeros(1, num_classes);
for i = 1:num_classes
    classes_binarias_reais = (classes_teste == i);
    classes_binarias_previstas = (classe_prevista == i);

    precisao = calcular_precisao(classes_binarias_reais, classes_binarias_previstas);
    recall = calcular_recall(classes_binarias_reais, classes_binarias_previstas);

    f1_scores(i) = 2 * (precisao * recall) / (precisao + recall + eps);
end

% F1-Score ponderado
contagem_classes = histc(classes_teste, 1:num_classes);
f1_score_ponderado = sum(f1_scores .* contagem_classes') / sum(contagem_classes);
disp(['F1-Score ponderado: ', num2str(f1_score_ponderado)]);


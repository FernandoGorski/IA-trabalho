
pkg load statistics;

% Colocar o endereço onde salvou
disp("Carregando 'Dados_Base'...");
dados_base = dlmread('xxx/IA-trabalho/Dados_Base.csv', ',');
disp("Dados_Base carregados com sucesso!");

disp("Carregando 'Classes_Base'...");
classes_base = dlmread('xxx/IA-trabalho/Classes_Base.csv', ',');
disp("Classes_Base carregados com sucesso!");

disp("Carregando 'Dados_Novos'...");
dados_novos = dlmread('xxx/IA-trabalho/Dados_Novos.csv', ',');
disp("Dados_Novos carregados com sucesso!");

X_ref = dados_base;       % Dados de entrada
y_ref = classes_base;     % Classes taxonômicas
X_sample = dados_novos;   % Dados a serem classificados

% ========================= Normalização =========================
disp("Normalizando os dados (Z-score)...");

mean_ref = mean(X_ref);
std_ref = std(X_ref);

X_ref = (X_ref - mean_ref) ./ std_ref;
X_sample = (X_sample - mean_ref) ./ std_ref;

disp("Normalização concluída!");

% ==================== Redução de Dimensionalidade ====================
disp("Aplicando PCA...");
[coeff, score, latent] = pca(X_ref);
k = 100;
X_ref_reduced = score(:, 1:k);
X_sample_reduced = (X_sample - mean(X_ref)) * coeff(:, 1:k);

% Visualização PCA
disp("Gerando visualização...");
scatter(score(:, 1), score(:, 2), 10, y_ref, 'filled');
title('Visualização PCA');
xlabel('PC1');
ylabel('PC2');
pause;

% ===================== Função para Classificação KNN ====================
function predicted_labels = knn_classify(X_test, X_train, y_train, k)

  num_test = size(X_test, 1);
  predicted_labels = zeros(num_test, 1);

  for i = 1:num_test
    distances = sqrt(sum((X_train - X_test(i, :)).^2, 2));

    [~, sorted_idx] = sort(distances);
    nearest_neighbors = y_train(sorted_idx(1:k));

    predicted_labels(i) = mode(nearest_neighbors);
  end
end

% ===================== Treinamento e Classificação ======================
disp("Classificando as amostras...");

k_value = 5;

predicted_labels = knn_classify(X_sample_reduced, X_ref_reduced, y_ref, k_value);

% ======================== Validação ============================
disp("Validando modelo...");

idx = randperm(length(y_ref));
train_idx = idx(1:floor(0.8*length(y_ref))); % 80% para treino
test_idx = idx(floor(0.8*length(y_ref)) + 1:end); % 20% para teste

X_train = X_ref_reduced(train_idx, :);
y_train = y_ref(train_idx);
X_test = X_ref_reduced(test_idx, :);
y_test = y_ref(test_idx);

y_pred_test = knn_classify(X_test, X_train, y_train, k_value);

% ===================== Cálculo de Métricas ==========================
TP = sum((y_pred_test == 1) & (y_test == 1)); % Verdadeiros positivos
TN = sum((y_pred_test == 0) & (y_test == 0)); % Verdadeiros negativos
FP = sum((y_pred_test == 1) & (y_test == 0)); % Falsos positivos
FN = sum((y_pred_test == 0) & (y_test == 1)); % Falsos negativos

% Acurácia
accuracy = (TP + TN) / (TP + TN + FP + FN);
disp(["Acurácia: ", num2str(accuracy)]);

% Sensibilidade
sensitivity = TP / (TP + FN);
disp(["Sensibilidade: ", num2str(sensitivity)]);

% Especificidade
specificity = TN / (TN + FP);
disp(["Especificidade: ", num2str(specificity)]);

% F1-Score
f1_score = 2 * (sensitivity * (TP / (TP + FP))) / (sensitivity + (TP / (TP + FP)));
disp(["F1-Score: ", num2str(f1_score)]);

% ==================== Validação Cruzada =======================
disp("Realizando Validação Cruzada...");
n_folds = 5;
cv_accuracies = zeros(1, n_folds);
for fold = 1:n_folds
    idx_fold = randperm(length(y_ref));
    train_idx = idx_fold(1:floor(0.8*length(y_ref))); % 80% para treino
    test_idx = idx_fold(floor(0.8*length(y_ref)) + 1:end); % 20% para teste

    X_train_cv = X_ref_reduced(train_idx, :);
    y_train_cv = y_ref(train_idx);
    X_test_cv = X_ref_reduced(test_idx, :);
    y_test_cv = y_ref(test_idx);

    y_pred_cv = knn_classify(X_test_cv, X_train_cv, y_train_cv, k_value);
    cv_accuracies(fold) = mean(y_pred_cv == y_test_cv) * 100;
end

mean_cv_accuracy = mean(cv_accuracies);
disp(["Acurácia média da validação cruzada: ", num2str(mean_cv_accuracy), "%"]);

% =================== Salvar Resultados =====================
disp("Salvando resultados...");
csvwrite('classified_samples.csv', predicted_labels);
disp("Concluído.");


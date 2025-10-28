Desvendando Padrões de Criptografia com Machine Learning

Este projeto é um classificador de padrões de criptografia desenvolvido em JavaScript, usando TensorFlow.js para a classificação e ScikitJS para uma regressão simples (ex.: descobrir o shift em ROT13). 
Ele permite que o usuário insira textos ou senhas criptografadas e o modelo identifica o tipo de criptografia utilizada, além de estimar valores numéricos quando aplicável.

Funcionalidades
--
Classificação automática de criptografias: plain, hex, base64, md5, sha1, sha256 e rot13.

Estimativa do shift em criptografias do tipo ROT usando regressão linear.

Front-end simples com HTML/CSS/JS para testar o modelo via navegador.

Treinamento de modelos reutilizáveis e salvamento para uso em produção.

--> COMO USAR

1. Instalar dependências

npm install
--
2. Gerar datasets

node gerar_dataset.js           # Dataset de classificação
node gerar_dataset_regressao.js # Dataset para regressão ROT
--
3. Treinar modelos

node train.js                  # Treinamento do classificador
node train_regressor_skit.js   # Treinamento da regressão linear
--
4. Rodar o servidor

node server.js
--

Acessar no navegador
Abra http://localhost:3000 E insira seu texto criptografado para ver o resultado da classificação e do shift (quando aplicável).
--

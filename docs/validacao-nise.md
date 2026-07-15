# Validação com exemplos do Nise

O núcleo matemático foi comparado com exemplos de *Engenharia de Sistemas de Controle*,
Norman S. Nise, 6ª edição. O arquivo usado na verificação possui SHA-256
`10DEBDB224E46AF232786CC02E34888A898132E5F265C24B65B46EE308DA3DAC`.

Os números de página abaixo identificam as páginas do arquivo PDF. O livro não é distribuído
com o projeto; o repositório contém somente os dados numéricos mínimos necessários para os
testes de regressão.

## Resultados verificados

| Referência | Grandezas do livro | Resultado do painel | Situação |
|---|---|---|---|
| Ex. 4.3, PDF 259 | `wn = 6 rad/s`, `zeta = 0,35` | `6 rad/s`, `0,35` | Aprovado |
| Ex. 4.5, PDF 267 | `Tp = 0,475 s`, `%UP = 2,838`, `Tr ~= 0,23 s` | `0,47514 s`, `2,83754%`, `0,22850 s` | Aprovado |
| Ex. 7.1, PDFs 511–512 | `e(infinito) = 1/2` | `0,5` | Aprovado |
| Ex. 8.5, PDF 604 | cruzamento em `K ~= 9,65`, `w ~= 1,59 rad/s` | `K = 9,64558`, `w = 1,58770 rad/s` | Aprovado |
| Ex. 10.7, PDFs 839–840 | estável para `K < 20`; marginal em `K = 20`, `w = sqrt(6)` | margem `20`; marginal em `K = 20`, `w = 2,44949 rad/s` | Aprovado |
| Ex. 10.9, PDFs 844–845 | resposta exata: ganho crítico `378` em `6,16 rad/s` | `378` em `6,16441 rad/s`; margem `19,5086 dB` | Aprovado |

## Conversão dos exemplos em malha fechada

Os Exemplos 4.3, 4.5 e 7.1 apresentam diretamente uma função de transferência em malha
fechada `T(s)`. Como o painel recebe a malha aberta `G(s)` e usa realimentação negativa
unitária, o teste fornece a função equivalente

```text
G(s) = T(s) / (1 - T(s)).
```

Assim, o `T(s) = G(s)/(1 + G(s))` calculado pelo programa reproduz exatamente o modelo do
livro, sem alterar o núcleo da aplicação.

## Diferença esperada no tempo de acomodação

No Exemplo 4.5, o livro usa a aproximação `Ts ~= 4/(zeta*wn)` e obtém `0,533 s`. O painel
aplica a definição numérica mais estrita: o primeiro instante após o qual a resposta permanece
dentro da banda de 2%. Para a mesma curva, o resultado é aproximadamente `0,574 s`.
O teste aceita a aproximação do livro e também verifica, ponto a ponto, que a resposta permanece
na banda depois do instante informado pelo painel.

## Como repetir

```powershell
python -m pytest tests/reference/test_nise_6e.py -v
```

Os dados de rastreabilidade estão em `tests/reference/nise_6e_cases.json`.

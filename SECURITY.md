# Segurança

## Segredos

Não versione tokens, credenciais RTSP, senhas de câmeras, API keys, arquivos `.env`, certificados ou URLs privadas. Use variáveis de ambiente e mantenha apenas `.env.example` no repositório.

## Webhooks e evidências

- considere o endpoint de webhook não confiável e use TLS/autenticação em produção;
- aplique políticas de retenção a snapshots e eventos;
- não envie vídeo bruto continuamente por padrão;
- revise imagens para dados pessoais e aplique redaction quando necessário.

## Relato de vulnerabilidade

Use o recurso privado de Security Advisories do GitHub deste repositório. Não publique credenciais, dados pessoais ou detalhes exploráveis em issues públicas.

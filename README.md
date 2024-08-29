# demo-ai-wandb

A handful of Weights and Biases programs that run on Red Hat Openshift.

This is work in progress.

###### Ephemeral Deployment on Openshift

Manual container deployment
```bash
oc new-project wandb
oc new-app wandb/local
```
Route
```bash
oc create route edge wandb --service=local --insecure-policy='Redirect' -n wandb
```

Visit the route and create a user/password.

How to retrieve the api key.
```
https://wandb-wandb.apps.ocp.sandbox2640.opentlc.com/authorize
```

Set the [wandb environment variables](https://docs.wandb.ai/guides/track/environment-variables) in a shell or workbench config.
```bash
export WANDB_API_KEY=local-api-key
export WANDB_BASE_URL=http://local.wandb:8080
```

```bash
wandb login --host http://local.wandb:8080
``'

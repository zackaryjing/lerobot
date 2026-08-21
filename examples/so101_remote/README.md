# SO101 remote inference

The WSL machine runs the CUDA policy server. The `jing` machine owns the SO101
follower and both cameras. An SSH reverse tunnel exposes the WSL server only as
`127.0.0.1:18080` on `jing`.

The model expected by this setup is:

`outputs/train/act_so101_test/checkpoints/025000/pretrained_model`

Open three WSL terminals, in this order:

```bash
examples/so101_remote/run.sh server
examples/so101_remote/run.sh tunnel
examples/so101_remote/run.sh client
```

Only start `client` after the follower is powered, the arm has clearance, and
someone is ready to cut power. Stop the client with Ctrl-C before stopping the
tunnel or server. The client configuration limits each commanded joint change
to 10 normalized position units and disables torque on disconnect.

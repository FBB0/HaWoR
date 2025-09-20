# ARCTIC Credentials Setup Guide

## Required Accounts

You need to register accounts on the following websites:

1. **ARCTIC Dataset**: https://arctic.is.tue.mpg.de/register.php
2. **SMPL-X**: https://smpl-x.is.tue.mpg.de/
3. **MANO**: https://mano.is.tue.mpg.de/

## Setting Up Credentials

After registering, export your credentials:

```bash
export ARCTIC_USERNAME=<YOUR_ARCTIC_EMAIL>
export ARCTIC_PASSWORD=<YOUR_ARCTIC_PASSWORD>
export SMPLX_USERNAME=<YOUR_SMPLX_EMAIL>
export SMPLX_PASSWORD=<YOUR_SMPLX_PASSWORD>
export MANO_USERNAME=<YOUR_MANO_EMAIL>
export MANO_PASSWORD=<YOUR_MANO_PASSWORD>
```

## Verify Credentials

Check if your credentials are set correctly:

```bash
echo $ARCTIC_USERNAME
echo $ARCTIC_PASSWORD
echo $SMPLX_USERNAME
echo $SMPLX_PASSWORD
echo $MANO_USERNAME
echo $MANO_PASSWORD
```

All should show your credentials (not empty).

## Next Steps

Once credentials are set, run:
```bash
python setup_arctic_integration.py --download-mini
```

This will download a small test dataset to verify everything works.

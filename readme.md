# meta vision core installation

to run the code you need to make an environment that has access to local site packages:

`python3 -m venv ~/eventcam_venv --system-site-packages
`
`source ~/eventcam_venv/bin/activate
`


```
# 0)
pip install h5py numpy matplotlib

# 1) Install curl if missing
sudo apt -y install curl

# 2) Add Prophesee GPG key
curl -L https://propheseeai.jfrog.io/artifactory/api/security/keypair/prophesee-gpg/public >/tmp/propheseeai.jfrog.op.asc
sudo cp /tmp/propheseeai.jfrog.op.asc /etc/apt/trusted.gpg.d

# 3) Add the OpenEB repository
sudo add-apt-repository 'https://propheseeai.jfrog.io/artifactory/openeb-debian/'

# 4) Update APT and install OpenEB
sudo apt update
sudo apt -y install metavision-openeb

```

packages used:

`build_cube_and_global_fft` generates fft file
`view_events` visualization of fft file
other files not used!



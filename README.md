# Joint PIC
This repository offers a joint channel estimation and symbol detection implementing the parallel interference cancellation method across platforms.

## How to install
* Install through `Matlab`
    * `HOME/Add-Ons/Get Add-Ons`: search `whatshow_phy_joint_pic` and install it.
* Install through `pip`
    ```sh
    pip install scipy
    pip install textremo-toolbox
    pip install textremo-phy-joint-pic
    ```
* After Installation
    * `Matlab`: run `init` to load all paths
    * `Python`: run `init.py` to test whether all required packages are installed. 
    
## How to use
* `CPE`: intially estimate the channel (`his`: delay varies slower than Doppler)
* `JPIC`: jointly estimate symbols and channel (`his`: delay varies slower than Doppler)

## Sample
- `VB`: variational Bayes
  - `testVbOtfsEmbedChe`: test VB on OTFS (Embed) channel estimation
  - `testVbOtfsEmbedCheSimp`: test VB on OTFS (Embed) channel estimation (simplified)

## Test
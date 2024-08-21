<!--
SPDX-FileCopyrightText: 2024 University of Rochester

SPDX-License-Identifier: CC-BY-SA-4.0
-->

## wgpu-runner

A simple benchmarking collector for webgpu kernels.


## Getting Started

You can run this locally with simple-server.py. You will need python3, npm, and node.

After cloning the repo, `cd` into its directory.  Then run `npm install`.  This will download typescript and the webgpu types.  Then, execute `npm run build`.  This is just an alias for `tsc`. If you are making changes to the code, you can also do `tsc --watch` and it will automatically rebuild the website with any changes.

Once the website has been built, you can use `simple-server.py` to serve its contents. Simply run `python3 simple-server.py`.  (You might have to change the port inside simple_server.py, default is 5030).

Then, open up your browser and navigate to `localhost:5030`

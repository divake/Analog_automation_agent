// CSVA Circuit Netlist Template
// Common-Source Voltage Amplifier with Resistive Load

// Simulator directives
simulator lang=spectre

// Include technology models
include "/path/to/models/technology.scs"

// Parameters
{{PARAMETERS}}

// Simulation specifications
{{SPECS}}

// Supply voltages
Vdd vdd 0 DC=1.8V
Vss vss 0 DC=0V

// Input signal
Vin in 0 AC 1 SIN(0.9V 10m 1Meg)

// Bias circuit (simplified)
Vbias bias 0 DC=0.9V
Rbias bias in 10k

// CSVA circuit
M1 out in vss vss nmos W=W L=L
Rd vdd out RD
Ibias vss vdd DC=ID

// Load capacitance (typically representing next stage)
Cl out 0 100f

// DC Analysis
dc dc start=0 stop=1.8 step=0.01

// AC Analysis for gain and bandwidth
ac ac start=1 stop=1G dec=10

// Transient Analysis
tran tran stop=5u step=0.01u

// Measurements
meas ac gain max vdb(out)
meas ac bw when vdb(out)='gain-3'
meas dc power avg 'abs(vdd*i(Vdd))'

// End of netlist 
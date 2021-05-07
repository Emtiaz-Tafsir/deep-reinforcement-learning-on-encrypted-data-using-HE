**plain_enc **: 
	Plain learning and encrypted evaluation.
	Works in two phases- learn_phase and test_phase.
	learn_phase range- 50 episodes by default.
	test_phase range- 10 episodes by default.
	contains three distinct neural networks.
	two for plain data- policy_net and target_net.
	one for encrypted data - policy_net_enc.
	policy_net- predict action on given input.
	target_net- for computing expected q value.
	policy_net_enc- predict action on given encrypted input.

**pure_enc **:
	Real time learning on encrypted data.
	Works in two modes- plain and encrypted.
	plain mode- test performance of the chosen model on plain data.
	encrypted mode- learn on encrypted data.
	contains two neural networks- policy_net and target_net.
	on encrypted mode, policy_net holds two internal networks.
	one for plain close-correlated dummy input data to generate neccessary backward function.
	one for encrypted actual input for generating output data.
	policy_net- predict action on given input.
	target_net- for computing expected q value.
	

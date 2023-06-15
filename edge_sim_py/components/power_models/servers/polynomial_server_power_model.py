import numpy as np
""" Contains a server power model definition."""


class PolynomialServerPowerModel:
    """Server power model that assumes a polynomial correlation between a server's power consumption and demand, based on real data from SPECPower benchmark.
    """

    @classmethod
    def get_power_consumption(cls, device: object) -> float:
        """Gets the power consumption of a server.

        Args:
            device (object): Server whose power consumption will be computed.

        Returns:
            power_consumption (float): Server's power consumption.
        """
        if "power_consumption" not in device.power_model_parameters:
            raise Exception("The power model parameters must contain the power_consumption parameter.")

        if device.active:
            utilization = np.array(np.arange(0, 1.1, 0.1))
            power = np.array(device.power_model_parameters["power_consumption"])
            fit_func = np.poly1d(np.polyfit(utilization, power, 7))

            demand = device.cpu_demand
            capacity = device.cpu
            utilization = demand / capacity

            power_consumption = fit_func(utilization)
        else:
            power_consumption = 0

        return power_consumption
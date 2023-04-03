""" Contains a method that defines the initial placement of services based on the well-known Worst-Fit heuristic."""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.container_image import ContainerImage
from edge_sim_py.components.container_layer import ContainerLayer
from edge_sim_py.components.service import Service
from edge_sim_py.components.user import User


def worst_fit_services():
    """Provisions services on the hosts with the largest amount of free resources that could accommodate them."""
    for service in Service.all():
        edge_servers = sorted(
            EdgeServer.all(),
            key=lambda s: ((s.cpu - s.cpu_demand) * (s.memory - s.memory_demand) * (s.disk - s.disk_demand)) ** (1 / 3),
            reverse=True,
        )

        for edge_server in edge_servers:
            # Checking if the host would have resources to host the service and its (additional) layers
            if edge_server.has_capacity_to_host(service=service):
                # Updating the host's resource usage
                edge_server.cpu_demand += service.cpu_demand
                edge_server.memory_demand += service.memory_demand

                # Creating relationship between the host and the registry
                service.server = edge_server
                edge_server.services.append(service)

                for layer_metadata in edge_server._get_uncached_layers(service=service):
                    layer = ContainerLayer(
                        digest=layer_metadata.digest,
                        size=layer_metadata.size,
                        instruction=layer_metadata.instruction,
                    )

                    # Updating host's resource usage based on the layer size
                    edge_server.disk_demand += layer.size

                    # Creating relationship between the host and the layer
                    layer.server = edge_server
                    edge_server.container_layers.append(layer)

                # Finding image provisioned on the edge server to get metadata
                template_image = ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)
                if template_image is None:
                    raise Exception(f"Could not find any container image with digest: {service.image_digest}")

                # Creating the new image on the target host
                image = ContainerImage()
                image.name = template_image.name
                image.digest = template_image.digest
                image.tag = template_image.tag
                image.architecture = template_image.architecture
                image.layers_digests = template_image.layers_digests

                # Connecting the new image to the target host
                image.server = edge_server
                edge_server.container_images.append(image)

                break

    # Calculating user communication paths and application delays
    for user in User.all():
        for application in user.applications:
            user.set_communication_path(app=application)

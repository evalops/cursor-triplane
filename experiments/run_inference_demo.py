import asyncio

from inference.serve import bootstrap_ray


async def main():
    controller = bootstrap_ray(["127.0.0.1:50051"], num_samplers=1)
    results = await controller.batch_rollouts.remote(["Fix failing unit tests"])
    print(results)


if __name__ == "__main__":
    asyncio.run(main())

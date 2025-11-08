#!/usr/bin/env python3
import argparse
import asyncio
from typing import Iterable

import grpc
from envd.generated import tools_pb2, tools_pb2_grpc


async def check_host(host: str, timeout: float = 3.0) -> bool:
    channel = grpc.aio.insecure_channel(host)
    stub = tools_pb2_grpc.ToolsStub(channel)
    try:
        await stub.ReadFile(tools_pb2.ReadFileReq(path="README.md"), timeout=timeout)
        return True
    except grpc.aio.AioRpcError:
        return False
    finally:
        await channel.close()


async def gather(hosts: Iterable[str]) -> None:
    tasks = [check_host(host) for host in hosts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for host, ok in zip(hosts, results, strict=False):
        status = "OK" if ok is True else "FAIL"
        print(f"{host}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Firecracker tool servers")
    parser.add_argument("hosts", nargs="+", help="Host:port pairs to probe")
    args = parser.parse_args()

    asyncio.run(gather(args.hosts))


if __name__ == "__main__":
    main()

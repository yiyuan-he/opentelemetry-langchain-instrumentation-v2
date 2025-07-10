import requests
from botocore import session, auth, awsrequest

SERVICE_LOGS = "logs"
SERVICE_XRAY = "xray"


class AwsAuthSession(requests.Session):

    def __init__(self, aws_region):
        self._aws_region = aws_region

        super().__init__()

    def request(
            self,
            method,
            url,
            data=None,
            headers=None,
            *args,
            **kwargs
    ):
        print("In AwsAuthSession.request")

        service = None
        if "xray" in url:
            service = SERVICE_XRAY
        elif "logs" in url:
            service = SERVICE_LOGS
        else:
            print("Error:: invalid service")

        botocore_session = session.Session()
        credentials = botocore_session.get_credentials()

        if credentials is not None:
            signer = auth.SigV4Auth(credentials, service, self._aws_region)

            request = awsrequest.AWSRequest(
                method=method,
                url=url,
                data=data,
                headers={"Content-Type": "application/x-protobuf"},
            )

            try:
                signer.add_auth(request)
                print("request.headers: ", request.headers)

                # update headers
                if headers is None:
                    headers = {}
                for key, value in request.headers.items():
                    headers[key] = value


            except Exception as signing_error:  # pylint: disable=broad-except
                print(signing_error)
                # _logger.error("Failed to sign request: %s", signing_error)

        return super().request(method, url, data=data, headers=headers, *args, **kwargs)

    def close(self):
        super().close()
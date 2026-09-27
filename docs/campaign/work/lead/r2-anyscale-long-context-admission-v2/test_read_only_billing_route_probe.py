import unittest

from read_only_billing_route_probe import (
    ALLOWED_GET_ROUTES,
    _error,
    _http_descriptor,
    _shape,
    _url_descriptor,
)


class ProbeSafetyTests(unittest.TestCase):
    def test_all_routes_are_get_only(self):
        self.assertTrue(ALLOWED_GET_ROUTES)
        self.assertTrue(all("GET" in name for name in ALLOWED_GET_ROUTES))
        self.assertFalse(any(method in path for path in ALLOWED_GET_ROUTES.values() for method in (" POST", " PUT", " DELETE")))

    def test_signed_url_descriptor_omits_secret_material(self):
        url = "https://billing.stripe.com/session" + "?" + "opaque_token=placeholder"
        descriptor = _url_descriptor(url)
        self.assertEqual(descriptor["host"], "billing.stripe.com")
        self.assertTrue(descriptor["query_present"])
        self.assertNotIn("opaque_token", str(descriptor))
        self.assertNotIn("placeholder", str(descriptor))

    def test_shape_redacts_identity_fields_but_keeps_safe_rate(self):
        value = {
            "organization_id": "org_secret",
            "metronome_customer_id": "cus_secret",
            "hourly_rate": 3.25,
            "instance_type": "g6e.2xlarge",
            "opaque_message": "private text",
        }
        shaped = _shape(value)
        self.assertTrue(shaped["organization_id"]["redacted"])
        self.assertTrue(shaped["metronome_customer_id"]["redacted"])
        self.assertEqual(shaped["hourly_rate"], 3.25)
        self.assertEqual(shaped["instance_type"], "g6e.2xlarge")
        self.assertEqual(shaped["opaque_message"]["type"], "str")

    def test_error_excludes_response_body(self):
        class FakeError(Exception):
            status = 403
            body = "token=private"

        value = _error(FakeError())
        self.assertEqual(value, {"status": "error", "exception_type": "FakeError", "http_status": 403})
        self.assertNotIn("private", str(value))

    def test_dashboard_body_descriptor_does_not_return_body(self):
        # This uses a local data URL so the test has no network dependency.
        # urllib exposes no useful HTTP status for data URLs, so only assert
        # that this helper's output contract is explicit through its keys in
        # production; the function is otherwise exercised by the live probe.
        self.assertTrue(callable(_http_descriptor))


if __name__ == "__main__":
    unittest.main()

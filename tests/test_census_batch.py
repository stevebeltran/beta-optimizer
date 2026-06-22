import io

from modules.census_batch import (
    build_census_staging,
    build_intersection_fallback_rows,
    make_census_batch_chunks,
)


class Upload(io.BytesIO):
    def __init__(self, name, text):
        super().__init__(text.encode("utf-8"))
        self.name = name


def test_census_staging_allows_city_state_addresses_without_zip():
    upload = Upload(
        "calls for service.csv",
        "\n".join(
            [
                "Incident,Nature,Incident address,City,State",
                "1,ASSIST,1227 S DEWEY ST,Eau Claire,WI",
                "2,HARASSMENT,SUMMER ST & HUEBSCH BLVD,Eau Claire,WI",
                "3,MISSING,,Eau Claire,WI",
                "4,BADSTATE,123 MAIN ST,Eau Claire,ZZ",
            ]
        ),
    )

    stage_df, _original_df, summary = build_census_staging([upload])

    assert summary["rows_total"] == 4
    assert summary["rows_ready"] == 1
    assert summary["intersection_rows"] == 1
    assert summary["rows_missing"] == 2
    assert summary["files"][0]["zip_col"] == ""

    chunks = make_census_batch_chunks(stage_df)
    assert len(chunks) == 1
    assert chunks[0]["rows"] == 1
    assert "1227 S DEWEY ST,Eau Claire,WI,," in chunks[0]["csv_bytes"].decode("utf-8")

    intersection_df = build_intersection_fallback_rows(stage_df)
    assert len(intersection_df) == 1
    assert intersection_df.iloc[0]["street"] == "SUMMER ST & HUEBSCH BLVD"


def test_cad_parser_does_not_treat_zip_as_coordinates():
    from modules.cad_parser import aggressive_parse_calls

    upload = Upload(
        "calls for service.csv",
        "\n".join(
            [
                "Nature,Incident address,City,State,zip",
                "ASSIST-EMS,1227 S DEWEY ST,Eau Claire,WI,54701",
                "HARASSMENT,SUMMER ST & HUEBSCH BLVD,Eau Claire,WI,54701",
                "CRASH-PROPERTY,500BLK GERMANIA ST; GERMANIA ST,Eau Claire,WI,54701",
            ]
        ),
    )

    parsed = aggressive_parse_calls([upload], require_valid_coordinates=True)

    assert parsed.empty

    upload = Upload(
        "calls for service.csv",
        "\n".join(
            [
                "Nature,Incident address,City,State,zip",
                "ASSIST-EMS,1227 S DEWEY ST,Eau Claire,WI,54701",
                "HARASSMENT,SUMMER ST & HUEBSCH BLVD,Eau Claire,WI,54701",
                "CRASH-PROPERTY,500BLK GERMANIA ST; GERMANIA ST,Eau Claire,WI,54701",
            ]
        ),
    )
    stage_df, _original_df, summary = build_census_staging([upload])

    assert summary["rows_ready"] == 2
    assert summary["intersection_rows"] == 1
    assert summary["files"][0]["zip_col"] == "zip"

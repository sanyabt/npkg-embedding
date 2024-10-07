'''
Load discovery patterns from database and return as dataframe
'''
import pandas as pd
import psycopg2 as pg, json
from sshtunnel import SSHTunnelForwarder

def connect_to_db():
    ##load creds
    creds = json.load(open("db_details.json", 'r'))
    ssh_tunnel = SSHTunnelForwarder(
    (creds['SSH_HOST'], int(creds['SSH_PORT'])),
    ssh_username=creds['PG_user'],
    ssh_private_key= creds['SSH_PKEY'],
    ssh_private_key_password = creds['PG_pass'],
    remote_bind_address=(creds['DB_HOST'], 5432),
    local_bind_address=(creds['DB_HOST'], 6543)
    )
    ssh_tunnel.start()
    try:
        conn = pg.connect(
        database = creds['PG_DB_NAME'],
        user = creds['PG_user'],
        host = ssh_tunnel.local_bind_host,
        port = ssh_tunnel.local_bind_port,
        password = creds["PG_pass"]
        )
    except Exception as e:
        print(e)
        print("Connection to DB failed")
        conn = None
    if not conn:
        print('Connection to DB failed')
    else:
        print('Connection to DB successful. Proceeding...')
    return conn

def load_discovery_patterns_1_pv(np_id, drug_id, adr_list, conn):
    cur = conn.cursor()
    pattern1_query = """select *
        from npkg_discovery_patterns.pv_signals_results_pattern1 dp
        where dp.d1 = %(np_concept_id)s and dp.d2 = %(drug_concept_id)s and dp.adr_omop_id = ANY(%(adr_list)s)
    """
    if conn:
        cur = conn.cursor()
        print('Extracting discovery pattern 1 results for PV signals')
        try:
            cur.execute(pattern1_query, {'np_concept_id': np_id, 'drug_concept_id': drug_id, 'adr_list': adr_list})
            print('Results fetched.')
            pattern1_results = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
            pattern1_results = pattern1_results.drop_duplicates()
            print('Number of rows fetched:', pattern1_results.shape[0])
        except (Exception, pg.DatabaseError) as e:
            print(e)
            print("Error: failed to fetch data from table")
            pattern1_results = None
            print('Results not fetched.')
            cur.close()
            conn.close()
    else:
        return None
    return pattern1_results

def load_edge_metadata_pattern1_pv(np_id, drug_id, conn):
    metadata_query = """
        with np_pattern as (
        select psrp.n_id, psrp.prot_id, psrp.drug_uri from npkg_discovery_patterns.pv_signals_results_pattern1 psrp 
        where psrp.d1 = %(np_concept_id)s and psrp.d2 = %(drug_concept_id)s
        )
        select * from npkg_discovery_patterns.edge_metadata_all ema 
        inner join np_pattern on ema.subject_uri = np_pattern.n_id and ema.object_uri = np_pattern.prot_id
        union 
        select * from npkg_discovery_patterns.edge_metadata_all ema 
        inner join np_pattern on ema.subject_uri = np_pattern.drug_uri and ema.object_uri = np_pattern.prot_id
        """
    cur = conn.cursor()
    print('Extracting edge metadata for discovery patterns results')
    try:
        cur.execute(metadata_query, {'np_concept_id': np_id, 'drug_concept_id': drug_id})
        print('Results fetched.')
        metadata_df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        metadata_df = metadata_df.drop_duplicates()
        print('Number of rows fetched:', metadata_df.shape[0])
    except (Exception, pg.DatabaseError) as e:
        print(e)
        print("Error: failed to fetch data from table")
        metadata_df = None
        print('Results not fetched.')
        cur.close()
        conn.close()
    return metadata_df

def get_omop_codes(adr_list, conn):
    meddra_codes_str = []
    for code in adr_list:
        meddra_codes_str.append(str(code))
    cur = conn.cursor()
    query_meddra_codes = """select distinct c.concept_id
    from staging_vocabulary.concept c
    where c.concept_code = ANY(%(adr_list)s)
    and c.vocabulary_id = 'MedDRA';
    """
    try:
        cur.execute(query_meddra_codes, {'adr_list': meddra_codes_str})
        omop_codes = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    except (Exception, pg.DatabaseError) as e:
        print(e)
        print("Error: failed to fetch data from table")
        omop_codes = None
    cur.close()
    if omop_codes is None:
        return None
    omop_codes_num = []
    omop_codes = omop_codes['concept_id'].tolist()
    for code in omop_codes:
        omop_codes_num.append(int(code))
    return omop_codes

def load_discovery_patterns_2_ref_data(np_uri, drug_uri, adr_list, conn):
    pass

def get_discovery_patterns_1_pv(np_id, drug_id, adr_list):
    conn = connect_to_db()
    ##get omop ids for ADRs
    omop_codes_num = get_omop_codes(adr_list, conn)

    pattern1_results = load_discovery_patterns_1_pv(np_id, drug_id, omop_codes_num, conn)
    if pattern1_results is None:
        return None
    pattern1_metadata = load_edge_metadata_pattern1_pv(np_id, drug_id, conn)
    if pattern1_metadata is not None:
        ##combine results and metadata
        pattern1_results['edge1_metadata'] = pd.NA
        pattern1_results['edge2_metadata'] = pd.NA
        pattern1_results['edge3_metadata'] = pd.NA
        for i in range(len(pattern1_results.index)):
            n_id = pattern1_results.at[i, 'n_id']
            prot_id = pattern1_results.at[i, 'prot_id']
            drug_uri = pattern1_results.at[i, 'drug_uri']
            edge1_data = pattern1_metadata[(pattern1_metadata['subject_uri'] == n_id) & (pattern1_metadata['object_uri'] == prot_id)]
            edge2_data = pattern1_metadata[(pattern1_metadata['subject_uri'] == drug_uri) & (pattern1_metadata['object_uri'] == prot_id)]
            if not edge1_data.empty:
                ##convert to dictionary with keys: pmid, sentence, section, pubtype, belief, year, also add parameters
                edge1_dict = edge1_data.to_dict(orient='records')[0]
                edge1_dict = {k: v for k, v in edge1_dict.items() if k in ['pmid', 'source_section', 'pubtype', 'belief', 'year',
                                                                           'dosage','area_conc_curve','max_conc','participants','plasma_conc_inc',
                                                                           'plasma_conc_dec','clearance','half_life','IC50','Ki','Km']}
                pattern1_results.at[i, 'edge1_metadata'] = edge1_dict
            if not edge2_data.empty:
                edge2_dict = edge2_data.to_dict(orient='records')[0]
                edge2_dict = {k: v for k, v in edge2_dict.items() if k in ['pmid', 'source_section', 'pubtype', 'belief', 'year',
                                                                           'dosage','area_conc_curve','max_conc','participants','plasma_conc_inc',
                                                                           'plasma_conc_dec','clearance','half_life','IC50','Ki','Km']}
                pattern1_results.at[i, 'edge2_metadata'] = edge2_dict
            ###edge3_data: if relation=correlated with, source = CTD, if relation=causes, source=onsides, if relation=treats, source=repodb
            edge_3_relation = pattern1_results.at[i, 'r3_id']
            if edge_3_relation == 'http://purl.obolibrary.org/obo/RO_0002610':
                edge3_dict = {'source': 'Comparative Toxicogenomic Database'}
            elif edge_3_relation == 'http://purl.obolibrary.org/obo/RO_0003302':
                edge3_dict = {'source': 'OnSIDES'}
            elif edge_3_relation == 'http://purl.obolibrary.org/obo/RO_0002606':
                edge3_dict = {'source': 'Drug Repositioning Database'}
            pattern1_results.at[i, 'edge3_metadata'] = edge3_dict
    if conn:
        conn.close()
    return pattern1_results

def get_discovery_patterns_ref_data_pattern2(np_uri, drug_uri, adr_list):
    pass